param([string]$DataDirectory=(Join-Path ([Environment]::GetFolderPath('Desktop')) 'Movira_test_Video'),[int]$Count=264)
$ErrorActionPreference='Stop'
# This script collects DATA only. It does not download code, APIs, or models.
$dataRoot=[IO.Path]::GetFullPath($DataDirectory)
New-Item -ItemType Directory -Path $dataRoot,(Join-Path $dataRoot 'images'),(Join-Path $dataRoot 'pixels') -Force | Out-Null
$annotationPath=Join-Path $dataRoot 'person_keypoints_val2017.json'
if(-not (Test-Path -LiteralPath $annotationPath)){
  $zipPath=Join-Path $dataRoot 'annotations_trainval2017.zip'
  Invoke-WebRequest 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip' -OutFile $zipPath
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $archive=[IO.Compression.ZipFile]::OpenRead($zipPath)
  try{[IO.Compression.ZipFileExtensions]::ExtractToFile($archive.GetEntry('annotations/person_keypoints_val2017.json'),$annotationPath,$true)}finally{$archive.Dispose()}
}
$coco=Get-Content -LiteralPath $annotationPath -Raw | ConvertFrom-Json -AsHashtable
$byImage=@{}
foreach($a in $coco.annotations){$key=[string]$a.image_id;if(-not $byImage.ContainsKey($key)){$byImage[$key]=[Collections.Generic.List[object]]::new()};$byImage[$key].Add($a)}
$selected=@($coco.images | Where-Object {$_.license -eq 4 -and @($byImage[[string]$_.id] | Where-Object {$_.num_keypoints -ge 12 -and $_.bbox[3] -ge 130 -and $_.bbox[2] -ge 40}).Count -gt 0} | Sort-Object id | Select-Object -First $Count)
Add-Type -AssemblyName System.Drawing
$records=[Collections.Generic.List[object]]::new();$manifest=[Collections.Generic.List[object]]::new()
$map=@(0,-1,5,6,7,8,9,10,-1,11,12,13,14,15,16,-1,-1,-1,-1)
foreach($info in $selected){
  $name=[string]$info.file_name;$imagePath=Join-Path (Join-Path $dataRoot 'images') $name
  $source="https://s3.amazonaws.com/images.cocodataset.org/val2017/$name"
  if(-not (Test-Path -LiteralPath $imagePath)){Invoke-WebRequest $source -OutFile $imagePath}
  $people=[Collections.Generic.List[object]]::new()
  foreach($a in $byImage[[string]$info.id]){
    if($a.iscrowd -or $a.num_keypoints -lt 1){continue}
    $points=@();$known=@()
    foreach($index in $map){
      if($index -lt 0){$points+=,$null;$known+=,$false;continue}
      $visibility=[int]$a.keypoints[$index*3+2];$known+=,($visibility -gt 0)
      if($visibility -gt 0){$points+=,@{x=[double]$a.keypoints[$index*3];y=[double]$a.keypoints[$index*3+1];occluded=($visibility -eq 1)}}else{$points+=,$null}
    }
    foreach($pair in @(@(1,2,3),@(8,9,10))){$index=$pair[0];$l=$points[$pair[1]];$r=$points[$pair[2]];if($null -ne $l -and $null -ne $r){$points[$index]=@{x=($l.x+$r.x)/2;y=($l.y+$r.y)/2;occluded=($l.occluded -or $r.occluded)};$known[$index]=$true}}
    $people.Add(@{box=@{x=[double]$a.bbox[0];y=[double]$a.bbox[1];w=[double]$a.bbox[2];h=[double]$a.bbox[3]};points=$points;known=$known;annotationId=$a.id})
  }
  $pixelsPath=Join-Path (Join-Path $dataRoot 'pixels') ($name+'.bgra')
  $bitmap=[Drawing.Bitmap]::new($imagePath)
  try{$rectangle=[Drawing.Rectangle]::new(0,0,$bitmap.Width,$bitmap.Height);$locked=$bitmap.LockBits($rectangle,[Drawing.Imaging.ImageLockMode]::ReadOnly,[Drawing.Imaging.PixelFormat]::Format32bppArgb)
    try{$pixels=[byte[]]::new($bitmap.Width*$bitmap.Height*4);for($row=0;$row -lt $bitmap.Height;$row++){[Runtime.InteropServices.Marshal]::Copy([IntPtr]::Add($locked.Scan0,$row*$locked.Stride),$pixels,$row*$bitmap.Width*4,$bitmap.Width*4)};[IO.File]::WriteAllBytes($pixelsPath,$pixels)}finally{$bitmap.UnlockBits($locked)}
  }finally{$bitmap.Dispose()}
  $origin=@{imageUrl=$source;originalFlickrUrl=$info.flickr_url;license='CC BY 2.0 (COCO metadata license id 4)';licenseUrl='https://creativecommons.org/licenses/by/2.0/';annotationLicense='CC BY 4.0';annotationSource='COCO Consortium, person_keypoints_val2017.json';derivedTargets='neck=shoulder midpoint; pelvis=hip midpoint; heel/toe NOT annotated';sha256=(Get-FileHash -LiteralPath $imagePath -Algorithm SHA256).Hash.ToLower()}
  $record=@{name=$name;width=$info.width;height=$info.height;people=$people.ToArray();source=$origin;dataURL='data:image/jpeg;base64,'+[Convert]::ToBase64String([IO.File]::ReadAllBytes($imagePath))}
  $records.Add($record)
  $manifest.Add(@{name=$name;imageId=$info.id;width=$info.width;height=$info.height;people=$people.Count;imageFile='images/'+$name;pixelFile='pixels/'+$name+'.bgra';source=$origin})
  Write-Output "$($records.Count)/$($selected.Count) $name ($($people.Count) people)"
}
@{format='movira-dataset-v1';records=$records.ToArray()} | ConvertTo-Json -Depth 20 -Compress | Set-Content -LiteralPath (Join-Path $dataRoot 'movira-dataset.json') -Encoding utf8
@{format='movira-collection-v1';collectedUtc=[DateTime]::UtcNow.ToString('o');annotationSha256=(Get-FileHash -LiteralPath $annotationPath -Algorithm SHA256).Hash.ToLower();selection="First $Count eligible by image id, CC BY id 4, >=12 annotated keypoints on a person >=130px high. Not a representative benchmark.";images=$manifest.ToArray()} | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $dataRoot 'sources.json') -Encoding utf8
Write-Output "Saved dataset: $dataRoot"
