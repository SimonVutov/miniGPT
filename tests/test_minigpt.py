from argparse import Namespace
import copy
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import torch
from torch.utils.data import DataLoader
import pytest
from model import GPT, GPTConfig, TokenizedDataset
from main import train_update, train, load_checkpoint, generate_text
from data import prepare

torch.set_num_threads(1)


def tiny_model(dropout=0):
    torch.manual_seed(5)
    return GPT(GPTConfig(8, 256, 1, 2, 16, dropout))


def test_causality_and_batch_independence():
    model = tiny_model().eval()
    tokens = torch.randint(0, 256, (2,8))
    original = model(tokens)
    changed = tokens.clone()
    changed[:,4:] = (changed[:,4:]+7)%256
    torch.testing.assert_close(model(changed)[:,:4], original[:,:4], atol=1e-6, rtol=1e-6)
    changed = tokens.clone()
    changed[1] = (changed[1]+12)%256
    torch.testing.assert_close(model(changed)[0], original[0], atol=1e-6, rtol=1e-6)
    changed = tokens.clone()
    changed[:,0] = (changed[:,0]+1)%256
    assert not torch.allclose(model(changed)[:,1:], original[:,1:])


@pytest.mark.parametrize("kwargs", [{"n_embd":15}, {"block_size":0}, {"dropout":1}, {"n_head":0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError): GPTConfig(**kwargs)


def test_generation_crops_context_and_restores_mode():
    model = tiny_model()
    prompt = torch.ones((1,20), dtype=torch.long)
    assert model.generate(prompt, 12, top_k=1).shape == (1,32)
    assert model.training
    with pytest.raises(ValueError): model.generate(prompt, temperature=0)
    with pytest.raises(ValueError): model.generate(prompt, top_k=257)
    with pytest.raises(ValueError): model(torch.ones((1,9),dtype=torch.long))
    with pytest.raises(ValueError): model(torch.ones((1,3)))


def test_accumulation_matches_large_batch():
    model = tiny_model()
    reference = copy.deepcopy(model)
    x = torch.randint(0,256,(5,8))
    y = torch.randint(0,256,(5,8))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    large_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    small_loss, count = train_update(model, optimizer, [(x[:3],y[:3]),(x[3:],y[3:])], torch.device("cpu"), scaler, clip=1e9)
    large_loss, _ = train_update(reference, large_optimizer, [(x,y)], torch.device("cpu"), scaler, clip=1e9)
    assert count == 40
    assert small_loss == pytest.approx(large_loss, abs=1e-6)
    for a,b in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)


def test_workers_do_not_repeat_shards(tmp_path):
    files=[]
    for i in range(4):
        path=tmp_path/f"{i}.pt"
        torch.save(torch.arange(i*40, i*40+21),path)
        files.append(path)
    dataset=TokenizedDataset(files,4,vocab_size=256)
    reference=Counter(tuple(x.tolist()) for x,_ in dataset)
    parallel=Counter(tuple(x[0].tolist()) for x,_ in DataLoader(dataset,batch_size=1,num_workers=2))
    assert reference == parallel
    assert len(reference) == 20
    assert all(count==1 for count in parallel.values())
    for x,y in dataset:
        assert torch.equal(x[1:],y[:-1])


def test_data_preparation_and_errors(tmp_path):
    directory=tmp_path/"data"
    metadata=prepare("abcdefghijklmnopqrstuvwxyz"*20,directory,shard_tokens=200)
    assert metadata["train_tokens"]+metadata["validation_tokens"] == 520
    assert sum(torch.load(p,weights_only=True).numel() for p in directory.glob("*/*.pt")) == 520
    with pytest.raises(ValueError): prepare("new text",directory)
    path=tmp_path/"invalid.pt"
    torch.save(torch.tensor([-1,2,3,4,5]),path)
    with pytest.raises(ValueError): list(TokenizedDataset([path],2,vocab_size=256))
    torch.save(torch.zeros((2,3)),path)
    with pytest.raises(ValueError): list(TokenizedDataset([path],2))
    with pytest.raises(ValueError): prepare("a",tmp_path/"short")


def arguments(data, output, steps=4, resume=None):
    return Namespace(data=data,output=output,steps=steps,resume=resume,device="cpu",precision="fp32",
                     seed=42,threads=1,batch_size=2,accumulation=3,workers=0,learning_rate=0.01,
                     block_size=8,layers=1,heads=2,embedding=16,dropout=0.1,eval_every=1,eval_batches=0)


def test_exact_resume_and_partial_accumulation(tmp_path):
    data=tmp_path/"data"
    prepare("abcdefghij"*9,data,validation_fraction=0.2)
    whole=train(arguments(data,tmp_path/"whole"))
    train(arguments(data,tmp_path/"split",steps=1))
    resumed=train(arguments(data,tmp_path/"split",resume=tmp_path/"split/last.pt"))
    full_model,full=load_checkpoint(tmp_path/"whole/last.pt",torch.device("cpu"))
    resumed_model,restored=load_checkpoint(tmp_path/"split/last.pt",torch.device("cpu"))
    for name,value in full_model.state_dict().items():
        torch.testing.assert_close(value,resumed_model.state_dict()[name],rtol=0,atol=0)
    assert whole["tokens_seen"] == resumed["tokens_seen"] == 128
    assert restored["state"]["step"] == 4
    assert whole["history"][-1]["validation_loss"] < whole["initial_validation_loss"]
    result=generate_text(tmp_path/"split/last.pt","abc",max_new_tokens=20,top_k=1)
    assert result.startswith("abc")
    invalid=arguments(data,tmp_path/"bad",steps=5,resume=tmp_path/"split/last.pt")
    invalid.accumulation=1
    with pytest.raises(ValueError,match="Resume requires"): train(invalid)


def test_cli_and_import_have_no_side_effects(tmp_path):
    root=Path(__file__).resolve().parents[1]
    code=f"import sys; sys.path.insert(0, {str(root)!r}); import data, main; assert callable(main.generate_text)"
    subprocess.run([sys.executable,"-c",code],cwd=tmp_path,check=True)
    assert not list(tmp_path.iterdir())
    result=subprocess.run([sys.executable,str(root/"main.py"),"train","--steps","0"],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode != 0 and "Invalid training counts" in result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_cuda_precision(precision):
    if precision=="bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 unavailable")
    model=tiny_model().cuda()
    optimizer=torch.optim.AdamW(model.parameters())
    scaler=torch.amp.GradScaler("cuda",enabled=precision=="fp16")
    x=torch.randint(0,256,(2,8))
    loss,count=train_update(model,optimizer,[(x,x)],torch.device("cuda"),scaler,precision)
    assert loss>0 and count==16


def test_cli_generation_with_legacy_stdout(monkeypatch, tmp_path):
    import io
    import main as cli
    output = io.BytesIO()
    stream = io.TextIOWrapper(output, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "argv", ["main.py", "generate", "--checkpoint", str(tmp_path/"unused.pt")])
    monkeypatch.setattr(cli, "generate_text", lambda *args, **kwargs: "sample: \ufffd\U0001f600")
    cli.main()
    stream.flush()
    assert output.getvalue() == b"sample: \\ufffd\\U0001f600\n"
