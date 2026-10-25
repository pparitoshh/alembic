"""Restore source aliases lost by global exact-chunk dedup, without adding content."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from markdownify import markdownify

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=Path('/Users/vaibhavmangroliya/Downloads/EU_AI_Factory_Documentation_2026-10-13')
ORIGINAL=ROOT/'runs/ai_factory_pilot_20261013/retrieval'
OUT=ROOT/'runs/retrieval_option_aware_20261014/retrieval'


def main():
    assert not OUT.exists(), 'Never overwrite prepared retrieval inputs'
    OUT.mkdir(parents=True)
    source=ORIGINAL/'documents.sqlite'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='5513283331ad05bf42b40aee94ff387546caa097b75f556842612caa03414577'
    shutil.copyfile(source,OUT/'documents.sqlite')
    tree=ast.parse((ROOT/'src/distillkit/seeds.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='chunk_text')
    scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_chunker','exec'),scope)
    chunk_text=scope['chunk_text']
    manifest=json.loads((ORIGINAL/'manifest.json').read_text())
    with sqlite3.connect('file:'+str(ARCHIVE/'catalog.sqlite')+'?mode=ro',uri=True) as catalog:
        records=[json.loads(r[0]) for r in catalog.execute("SELECT metadata FROM records WHERE status='downloaded' AND source IN ('slurm_upstream','lumi_docs','cineca_hpc')")]
    by_source={(r.get('resolved_url') or r['url'],r['sha256']):r for r in records}
    db=sqlite3.connect(OUT/'documents.sqlite')
    original_rows=db.execute('SELECT * FROM chunks ORDER BY id').fetchall()
    chunks={hashlib.sha256(row[4].encode()).hexdigest():row[0] for row in original_rows}
    assert len(chunks)==len(original_rows)
    db.execute('CREATE TABLE source_aliases(chunk_id INTEGER,url TEXT,site TEXT,title TEXT,original_sha256 TEXT,retrieved_at TEXT,UNIQUE(chunk_id,url))')
    bindings=[]
    for document in manifest['source_documents']:
        key=(document['url'],document['sha256'])
        assert key in by_source, 'Original archived source missing: '+document['url']
        r=by_source[key];raw=(ARCHIVE/r['original_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==document['sha256']
        soup=BeautifulSoup(raw,'lxml')
        selectors={'slurm_upstream':'div.content','lumi_docs':'article.md-content__inner','cineca_hpc':'[role=main]'}
        main=soup.select_one(selectors[r['source_id']]);assert main is not None
        for el in main.select('script,style,nav,footer,#cse,.headerlink,.linenos,.md-content__button,.md-source-file'):
            el.decompose()
        code=[]
        for pre in main.select('pre'):
            block=pre.get_text().strip();token='DKCODEBLOCK%05dTOKEN'%len(code)
            code.append(block);p=soup.new_tag('p');p.string=token;pre.replace_with(p)
        for a in main.select('a[href]'):a['href']=urljoin(document['url'],a['href'])
        text=markdownify(str(main),heading_style='ATX',bullets='-',strip=['img'])
        for i,block in enumerate(code):
            fence='`'*max(3,1+max([0]+[len(m.group()) for m in re.finditer(r'`+',block)]))
            text=text.replace('DKCODEBLOCK%05dTOKEN'%i,'\n\n'+fence+'\n'+block+'\n'+fence+'\n\n')
        added=0
        for chunk in chunk_text(text,2600):
            if not 200<=len(chunk)<=7000:continue
            sha=hashlib.sha256(chunk.encode()).hexdigest()
            assert sha in chunks, 'Content differs from frozen index: '+document['url']
            db.execute('INSERT OR IGNORE INTO source_aliases VALUES(?,?,?,?,?,?)',
                       (chunks[sha],document['url'],document['site'],r.get('title',''),r['sha256'],r['retrieved_at']))
            added+=1
        bindings.append({'url':document['url'],'original_sha256':r['sha256'],'reconstructed_chunks':added})
    assert db.execute('SELECT * FROM chunks ORDER BY id').fetchall()==original_rows
    missing=db.execute('SELECT count(*) FROM chunks c WHERE NOT EXISTS(SELECT 1 FROM source_aliases a WHERE a.chunk_id=c.id AND a.url=c.url)').fetchone()[0]
    assert missing==0, 'Lost original source binding'
    db.execute('CREATE INDEX source_alias_url ON source_aliases(url,site,chunk_id)')
    count=db.execute('SELECT count(*) FROM source_aliases').fetchone()[0]
    sbatch=db.execute("SELECT count(*) FROM source_aliases WHERE url='https://slurm.schedmd.com/sbatch.html'").fetchone()[0]
    db.commit();assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok';db.close()
    report={'purpose':'retrieval_only_source_attribution_repair_not_SFT','source_documents':len(bindings),
            'unique_content_chunks_unchanged':len(original_rows),'source_aliases':count,
            'sbatch_source_chunks_with_aliases':sbatch,'original_documents_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'new_documents_sha256':hashlib.sha256((OUT/'documents.sqlite').read_bytes()).hexdigest(),
            'content_and_fts_index_unchanged':True,'questions_or_reference_answers_indexed':False,
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bindings':bindings}
    (OUT/'alias_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()
