"""I/O-only regressions; no BPI data, training or Torch dependency required."""
import ast,errno,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('exp05_io_under_test',ROOT/'src/ipat/experiments/exp05_io.py')
io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io)


def winerror(code):
    error=PermissionError(errno.EACCES,'simulated Windows denial');error.winerror=code;return error


class IOTests(unittest.TestCase):
    def test_success_closes_handle_before_replace(self):
        with tempfile.TemporaryDirectory() as t:
            target=Path(t)/'x.json';target.write_bytes(b'old');seen=[];replace=io.os.replace
            def writer(f):seen.append(f);f.write(b'new')
            def swap(src,dst):self.assertTrue(seen[0].closed);replace(src,dst)
            with patch.object(io.os,'replace',side_effect=swap):io.atomic_write(target,writer)
            self.assertEqual(target.read_bytes(),b'new');self.assertEqual(list(Path(t).glob('*.tmp')),[])
    def test_temporary_windows_denials_recover(self):
        for code in (5,32,33):
            with self.subTest(code=code),tempfile.TemporaryDirectory() as t:
                target=Path(t)/'x';target.write_bytes(b'old');replace=io.os.replace;calls=[];writes=[]
                def swap(src,dst):
                    calls.append(str(src));self.assertEqual(target.read_bytes(),b'old')
                    if len(calls)<3:raise winerror(code)
                    replace(src,dst)
                def writer(f):writes.append(1);f.write(b'new')
                with patch.object(io.os,'replace',side_effect=swap),patch.object(io.time,'sleep') as sleep:
                    io.atomic_write(target,writer);self.assertEqual(sleep.call_count,2)
                self.assertEqual(target.read_bytes(),b'new');self.assertEqual(len(writes),1);self.assertEqual(len(set(calls)),1)
    def test_exhaustion_preserves_old_and_new(self):
        with tempfile.TemporaryDirectory() as t:
            target=Path(t)/'x';target.write_bytes(b'old')
            with patch.object(io.os,'replace',side_effect=winerror(5)) as replace,patch.object(io.time,'sleep') as sleep:
                with self.assertRaises(PermissionError) as raised:io.atomic_write(target,lambda f:f.write(b'new'),attempts=3)
            self.assertEqual(replace.call_count,3);self.assertEqual(sleep.call_count,2)
            self.assertEqual(target.read_bytes(),b'old');self.assertEqual(next(Path(t).glob('*.tmp')).read_bytes(),b'new')
            self.assertIn('Complete candidate retained',raised.exception.__notes__[0])
    def test_unrelated_error_fails_without_retry(self):
        with tempfile.TemporaryDirectory() as t,patch.object(io.os,'replace',side_effect=OSError(errno.ENOSPC,'full')) as replace,patch.object(io.time,'sleep') as sleep:
            with self.assertRaises(OSError):io.atomic_write(Path(t)/'x',lambda f:f.write(b'new'))
            self.assertEqual(replace.call_count,1);sleep.assert_not_called()
    def test_serialisation_failure_preserves_old(self):
        with tempfile.TemporaryDirectory() as t:
            target=Path(t)/'x';target.write_bytes(b'old')
            def fail(f):f.write(b'partial');raise ValueError('serializer')
            with self.assertRaises(ValueError),patch.object(io.os,'replace') as replace:io.atomic_write(target,fail)
            replace.assert_not_called();self.assertEqual(target.read_bytes(),b'old')
    def test_distinct_temporary_names(self):
        with tempfile.TemporaryDirectory() as t,patch.object(io.os,'replace',side_effect=winerror(5)):
            for _ in range(2):
                with self.assertRaises(PermissionError):io.atomic_write(Path(t)/'x',lambda f:f.write(b'new'),attempts=1)
            self.assertEqual(len(list(Path(t).glob('*.tmp'))),2)
    def test_delay_is_bounded(self):
        with tempfile.TemporaryDirectory() as t,patch.object(io.os,'replace',side_effect=winerror(5)),patch.object(io.time,'sleep') as sleep:
            with self.assertRaises(PermissionError):io.atomic_write(Path(t)/'x',lambda f:f.write(b'new'))
            self.assertEqual([c.args[0] for c in sleep.call_args_list],[.1,.2,.4,.8,1.,1.,1.,1.,1.])
    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):io.atomic_write('unused',lambda f:None,attempts=0)
    def wrappers(self):
        # Execute the actual two wrapper definitions only; do not fake a Torch run.
        module=ast.parse((ROOT/'src/ipat/experiments/exp05.py').read_text())
        selected=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ('write_json','save_checkpoint')]
        ns={'json':json,'atomic_write':Mock(),'torch':Mock()};exec(compile(ast.Module(body=selected,type_ignores=[]),'wrappers','exec'),ns);return ns
    def test_json_wrapper_bytes_and_nonfinite_rejection(self):
        ns=self.wrappers();ns['write_json']('x',{'a':'é'});handle=Mock();ns['atomic_write'].call_args.args[1](handle)
        self.assertEqual(json.loads(handle.write.call_args.args[0]),{'a':'é'})
        ns['atomic_write'].reset_mock()
        with self.assertRaises(ValueError):ns['write_json']('x',{'a':float('nan')})
        ns['atomic_write'].assert_not_called()
    def test_checkpoint_wrapper_uses_shared_writer(self):
        ns=self.wrappers();value={'test':1};ns['save_checkpoint']('x',value);handle=Mock();ns['atomic_write'].call_args.args[1](handle)
        ns['torch'].save.assert_called_once_with(value,handle)

if __name__=='__main__':unittest.main()
