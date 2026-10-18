import sqlite3
import tempfile
from pathlib import Path
import unittest
from retrieve import command_names, retrieve


class RetrievalTests(unittest.TestCase):
    def test_only_actual_command_names(self):
        self.assertEqual(command_names('Using `sbatch`, inspect with sacct then sbatch.'), ['sbatch','sacct'])
        self.assertEqual(command_names('batch processing with control nodes'), [])

    def test_command_context_and_site_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'docs.sqlite'
            with sqlite3.connect(p) as db:
                db.execute('CREATE TABLE chunks(id INTEGER PRIMARY KEY,url TEXT,site TEXT,title TEXT,text TEXT,retrieved_at TEXT)')
                db.execute('CREATE VIRTUAL TABLE search USING fts5(text,title,url,content=chunks,content_rowid=id)')
                rows=[(1,'https://slurm.schedmd.com/srun.html','Slurm','srun','GPU GPU GPU memory memory nodes sbatch submit','now'),
                      (2,'https://slurm.schedmd.com/sbatch.html','Slurm','sbatch','--mem memory per node','now'),
                      (3,'https://slurm.schedmd.com/sbatch.html','Slurm','sbatch','--gpus GPUs per job','now'),
                      (4,'https://docs.lumi-supercomputer.eu/jobs/','LUMI','LUMI','LUMI sbatch GPU memory policy','now')]
                db.executemany('INSERT INTO chunks VALUES(?,?,?,?,?,?)',rows)
                db.execute('INSERT INTO search(rowid,text,title,url) SELECT id,text,title,url FROM chunks')
            docs=retrieve(p,'submit GPU memory nodes using sbatch')
            self.assertEqual([d['id'] for d in docs[:2]], [2,3])
            self.assertEqual(len({d['id'] for d in docs}),3)
            self.assertTrue(all(d['site']=='Slurm' for d in docs))
            self.assertTrue(all(d['site']=='LUMI' for d in retrieve(p,'LUMI sbatch GPU memory')))


if __name__=='__main__':unittest.main()
