/* Add provenance and ignore regions using data, not external executable code. */
const fs=require('node:fs'),path=require('node:path');
const root=path.resolve(process.argv[2]||path.join(process.env.USERPROFILE,'Desktop','Movira_test_Video'));
const read=name=>JSON.parse(fs.readFileSync(path.join(root,name),'utf8').replace(/^\uFEFF/,''));
async function run(){
  const dataset=read('movira-dataset.json'),manifest=read('sources.json'),annotations=read('person_keypoints_val2017.json'),ignore=new Map();
  for(const a of annotations.annotations){if(!ignore.has(a.image_id))ignore.set(a.image_id,[]);ignore.get(a.image_id).push({x:a.bbox[0],y:a.bbox[1],w:a.bbox[2],h:a.bbox[3]});}
  let index=0,failures=0;
  async function work(){while(index<dataset.records.length){const at=index++,r=dataset.records[at];r.ignoreBoxes=ignore.get(Number(r.name.split('.')[0]))||[];
    try{const match=String(r.source.originalFlickrUrl).match(/\/(\d+)_/);if(!match)throw Error('No photo id');const page=await fetch('https://www.flickr.com/photo.gne?id='+match[1],{signal:AbortSignal.timeout(15000)});if(!page.ok)throw Error('Source page unavailable');const canonical=page.url;await page.body.cancel();const metadata=await fetch('https://www.flickr.com/services/oembed/?format=json&url='+encodeURIComponent(canonical),{signal:AbortSignal.timeout(15000)});if(!metadata.ok)throw Error('Source author metadata unavailable');const data=await metadata.json();Object.assign(r.source,{title:data.title,author:data.author_name,authorUrl:data.author_url,flickrPage:canonical,creditVerifiedUtc:new Date().toISOString()});}
    catch(error){r.source.creditStatus='Original author page metadata could not be confirmed automatically';failures++;}
    const item=manifest.images.find(item=>item.name===r.name);item.source=r.source;if((at+1)%20===0)console.log(`Credits ${at+1}/${dataset.records.length}`);
  }}
  await Promise.all(Array.from({length:6},work));
  manifest.selection=`First ${dataset.records.length} eligible images by id; CC BY 2.0 metadata, >=12 joints on a person >=130px high.`;
  fs.writeFileSync(path.join(root,'movira-dataset.json'),JSON.stringify(dataset));fs.writeFileSync(path.join(root,'sources.json'),JSON.stringify(manifest,null,2));
  console.log(JSON.stringify({images:dataset.records.length,people:dataset.records.reduce((n,r)=>n+r.people.length,0),unconfirmedCredits:failures}));
}
run().catch(error=>{console.error(error.message);process.exitCode=1;});
