param([ValidateSet('Deploy','Status')][string]$Action='Status')
$ErrorActionPreference='Stop'
$env:GIT_TERMINAL_PROMPT='0';$env:GCM_INTERACTIVE='Never'
if($Action -eq 'Deploy'){
  node --check web/app.js
  if($LASTEXITCODE -ne 0){throw 'App syntax check failed'}
  node --check web/engine.js
  if($LASTEXITCODE -ne 0){throw 'Engine syntax check failed'}
  node --test tests/engine.test.cjs
  if($LASTEXITCODE -ne 0){throw 'Tests failed'}
  if(git status --porcelain){throw 'Commit the reviewed source before deploying'}
  $siteCommit=git subtree split --prefix=web HEAD
  if($LASTEXITCODE -ne 0){throw 'Site source preparation failed'}
  git push origin "${siteCommit}:refs/heads/movira-site"
  if($LASTEXITCODE -ne 0){throw 'Site push failed; no forced update was attempted'}
}
# Read the already-authorized Git credential in memory, never log it.
$credentialLines="protocol=https`nhost=github.com`npath=yumin0117/Movira.git`n`n" | git credential fill
if($LASTEXITCODE -ne 0){throw 'Saved Git credential unavailable'}
$credential=@{};foreach($line in $credentialLines){$pair=$line -split '=',2;if($pair.Count -eq 2){$credential[$pair[0]]=$pair[1]}}
if(-not $credential.password){throw 'No GitHub credential was returned'}
$headers=@{Authorization="Bearer $($credential.password)";Accept='application/vnd.github+json';'X-GitHub-Api-Version'='2026-03-10';'User-Agent'='MoviraOwnBuild'}
$api='https://api.github.com/repos/yumin0117/Movira/pages'
if($Action -eq 'Deploy'){
  $body=@{build_type='legacy';source=@{branch='movira-site';path='/'}} | ConvertTo-Json -Compress
  $site=Invoke-RestMethod -Method Put -Uri $api -Headers $headers -ContentType 'application/json' -Body $body
  [ordered]@{siteCommit=$siteCommit;url=$site.html_url;buildType=$site.build_type;branch=$site.source.branch} | ConvertTo-Json
}else{
  $site=Invoke-RestMethod -Uri $api -Headers $headers
  $build=Invoke-RestMethod -Uri ($api+'/builds/latest') -Headers $headers
  [ordered]@{url=$site.html_url;branch=$site.source.branch;status=$build.status;commit=$build.commit;error=$build.error.message;updated=$build.updated_at} | ConvertTo-Json
}
