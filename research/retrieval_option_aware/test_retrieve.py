import sqlite3
import tempfile
from pathlib import Path
import unittest
from retrieve import definitions, retrieve, facet_options, resource_request


class OptionRetrievalTests(unittest.TestCase):
    def test_prose_mentions_are_not_definitions(self):
        self.assertEqual(definitions('See **--mem** for memory.\n**-N**, **--nodes**=<n>\n**--mem**=<size>'), {'nodes','mem'})

    def test_memory_scope_is_preserved(self):
        self.assertEqual(facet_options('16 GiB of memory per node')['memory'], {'mem'})
        self.assertEqual(facet_options('4 GB per CPU')['memory'], {'mem-per-cpu'})
        self.assertEqual(facet_options('memory per GPU')['memory'], {'mem-per-gpu'})

    def test_workload_context_is_not_resource_request(self):
        self.assertFalse(resource_request('My GPU application failed. How do I configure sbatch email?', 'sbatch'))
        self.assertFalse(resource_request('My 4 GPU job failed. Inspect it with sacct.', 'sacct'))
        self.assertTrue(resource_request('Request 4 GPUs and 32 GB per node using sbatch.', 'sbatch'))

    def test_memory_definition_is_not_crowded_out_by_repeated_gpu_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'docs.sqlite'
            with sqlite3.connect(p) as db:
                db.execute('CREATE TABLE chunks(id INTEGER PRIMARY KEY,url TEXT,site TEXT,title TEXT,text TEXT,retrieved_at TEXT)')
                db.execute('CREATE VIRTUAL TABLE search USING fts5(text,title,url,content=chunks,content_rowid=id)')
                rows=[(1,'https://slurm.schedmd.com/sbatch.html','Slurm','sbatch','GPU GPU GPU memory nodes GPU configuration.','now'),
                      (2,'https://slurm.schedmd.com/sbatch.html','Slurm','sbatch','**--mem**=<size>\nMemory required per node.','now'),
                      (3,'https://slurm.schedmd.com/srun.html','Slurm','srun','**--gpus**=<n>\nGPUs per job.','now'),
                      (4,'https://slurm.schedmd.com/sbatch.html','Slurm','sbatch','**-N**, **--nodes**=<n>\nNodes required.','now'),
                      (5,'https://docs.lumi-supercomputer.eu/jobs/','LUMI','LUMI','LUMI sbatch GPU memory','now')]
                db.executemany('INSERT INTO chunks VALUES(?,?,?,?,?,?)',rows)
                db.execute('INSERT INTO search(rowid,text,title,url) SELECT id,text,title,url FROM chunks')
                db.execute('CREATE TABLE source_aliases(chunk_id INTEGER,url TEXT,site TEXT,title TEXT,original_sha256 TEXT,retrieved_at TEXT)')
                db.executemany('INSERT INTO source_aliases VALUES(?,?,?,?,?,?)',
                               [(r[0],'https://slurm.schedmd.com/sbatch.html','Slurm','sbatch','sourcehash','now') for r in rows[:4]])
            result=retrieve(p,'sbatch job with 4 GPUs, 32 GB memory and 2 nodes')
            self.assertEqual({r['id'] for r in result},{2,3,4})
            self.assertEqual(len(result),3)
            self.assertTrue(all(r['url']=='https://slurm.schedmd.com/sbatch.html' for r in result))
            self.assertTrue(all(r['site']=='LUMI' for r in retrieve(p,'LUMI sbatch memory GPUs')))


if __name__=='__main__':unittest.main()
