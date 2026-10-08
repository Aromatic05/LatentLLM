"""Generate single-word, importable Python protocol modules from transfer.proto."""
from pathlib import Path
from tempfile import TemporaryDirectory

from grpc_tools import protoc

root = Path(__file__).resolve().parents[1]
target = root / 'server/src/latentllm/transport/schema'
target.mkdir(parents=True, exist_ok=True)
with TemporaryDirectory() as tmp:
    result = protoc.main([
        'protoc', f'-I{root / "proto"}', f'--python_out={tmp}',
        f'--grpc_python_out={tmp}', str(root / 'proto/transfer.proto'),
    ])
    if result:
        raise SystemExit(result)
    model = Path(tmp, 'transfer_pb2.py').read_text()
    service = Path(tmp, 'transfer_pb2_grpc.py').read_text()
    service = service.replace('import transfer_pb2 as transfer__pb2', 'from . import model as transfer__pb2')
    (target / 'model.py').write_text(model)
    (target / 'service.py').write_text(service)
    (target / '__init__.py').touch()
print(f'Protocol generated in {target}')
