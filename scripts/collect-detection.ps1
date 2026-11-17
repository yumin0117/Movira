param([int]$PeopleImages=1000,[int]$BackgroundImages=160,[string]$DataDirectory=(Join-Path ([Environment]::GetFolderPath('Desktop')) 'Movira_test_Video'),[ValidateSet('train2017','val2017')][string]$Split='train2017',[string]$Subfolder='detection-train',[switch]$ExcludeExisting,[int]$MinimumKeypoints=5,[int]$MinimumHeight=40,[int]$MinimumWidth=16)
$ErrorActionPreference='Stop';$root=[IO.Path]::GetFullPath($DataDirectory);$destination=Join-Path $root $Subfolder
New-Item -ItemType Directory -Path $destination,(Join-Path $destination 'images'),(Join-Path $destination 'pixels') -Force | Out-Null
$annotation=Join-Path $root ('person_keypoints_'+$Split+'.json')
if(-not (Test-Path -LiteralPath $annotation)){$zipFile=Join-Path $root 'annotations_trainval2017.zip';if(-not (Test-Path -LiteralPath $zipFile)){Invoke-WebRequest 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip' -OutFile $zipFile};Add-Type -AssemblyName System.IO.Compression.FileSystem;$archive=[IO.Compression.ZipFile]::OpenRead($zipFile);try{[IO.Compression.ZipFileExtensions]::ExtractToFile($archive.GetEntry('annotations/person_keypoints_'+$Split+'.json'),$annotation,$false)}finally{$archive.Dispose()}}
$coco=Get-Content -LiteralPath $annotation -Raw | ConvertFrom-Json -AsHashtable;$groups=@{}
foreach($a in $coco.annotations){$key=[string]$a.image_id;if(-not $groups.ContainsKey($key)){$groups[$key]=[Collections.Generic.List[object]]::new()};$groups[$key].Add($a)}
$eligible=@($coco.images | Where-Object {$_.license -eq 4} | Sort-Object {([long]$_.id*2654435761)%4294967296})
if($ExcludeExisting){$eligible=@($eligible | Where-Object {-not (Test-Path -LiteralPath (Join-Path (Join-Path $root 'images') $_.file_name))})}
$positive=@($eligible | Where-Object {@($groups[[string]$_.id] | Where-Object {-not $_.iscrowd -and $_.num_keypoints -ge $MinimumKeypoints -and $_.bbox[3] -ge $MinimumHeight -and $_.bbox[2] -ge $MinimumWidth}).Count -gt 0} | Select-Object -First $PeopleImages)
$negative=@($eligible | Where-Object {-not $groups.ContainsKey([string]$_.id)} | Select-Object -First $BackgroundImages)
$selected=@($positive)+@($negative);$work=@()
foreach($info in $selected){$all=@($groups[[string]$info.id]);$people=@($all | Where-Object {-not $_.iscrowd -and $_.num_keypoints -ge $MinimumKeypoints -and $_.bbox[3] -ge $MinimumHeight -and $_.bbox[2] -ge $MinimumWidth} | ForEach-Object {@{box=@{x=$_.bbox[0];y=$_.bbox[1];w=$_.bbox[2];h=$_.bbox[3]};keypoints=$_.keypoints;id=$_.id}});$work+=,@{name=$info.file_name;id=$info.id;width=$info.width;height=$info.height;people=$people;ignoreBoxes=@($all | Where-Object {$_} | ForEach-Object {@{x=$_.bbox[0];y=$_.bbox[1];w=$_.bbox[2];h=$_.bbox[3]}});source=@{imageUrl="https://s3.amazonaws.com/images.cocodataset.org/$Split/$($info.file_name)";originalFlickrUrl=$info.flickr_url;license='CC BY 2.0, COCO metadata id 4';annotationLicense='COCO Consortium CC BY 4.0'}}}
$records=@($work | ForEach-Object -Parallel {
  $item=$_;$folder=$using:destination;$ErrorActionPreference='Stop';$file=Join-Path (Join-Path $folder 'images') $item.name;$raw=Join-Path (Join-Path $folder 'pixels') ($item.name+'.bgra')
  try{
    if(-not (Test-Path -LiteralPath $file)){Invoke-WebRequest -Uri $item.source.imageUrl -OutFile $file -TimeoutSec 45}
    if(-not (Test-Path -LiteralPath $raw)){
      Add-Type -AssemblyName System.Drawing;$bitmap=[Drawing.Bitmap]::new($file)
      try{$locked=$bitmap.LockBits([Drawing.Rectangle]::new(0,0,$bitmap.Width,$bitmap.Height),[Drawing.Imaging.ImageLockMode]::ReadOnly,[Drawing.Imaging.PixelFormat]::Format32bppArgb);try{$bytes=[byte[]]::new($bitmap.Width*$bitmap.Height*4);for($row=0;$row -lt $bitmap.Height;$row++){[Runtime.InteropServices.Marshal]::Copy([IntPtr]::Add($locked.Scan0,$row*$locked.Stride),$bytes,$row*$bitmap.Width*4,$bitmap.Width*4)};[IO.File]::WriteAllBytes($raw,$bytes)}finally{$bitmap.UnlockBits($locked)}}finally{$bitmap.Dispose()}
    }
    $item.source.sha256=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLower();$item
  }catch{Write-Warning ($item.name+': '+$_.Exception.Message)}
} -ThrottleLimit 8)
$records=@($records | Sort-Object {([long]$_.id*2654435761)%4294967296});@{format='movira-detection-data-v1';source="COCO $Split, data only, no code or weights";records=$records} | ConvertTo-Json -Depth 20 -Compress | Set-Content -LiteralPath (Join-Path $destination 'manifest.json') -Encoding utf8
Write-Output "Collected $($records.Count) images, $((@($records | ForEach-Object {$_.people})).Count) person annotations in $destination"
