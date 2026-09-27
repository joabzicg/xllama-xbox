#Requires -Version 5.1
<# Fail closed if the Xbox vision build drifts from the audited llama.cpp pin.
   This checks build inputs, not successful MSVC compilation or console runtime. #>
param()
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$llama = Join-Path $repo 'llama.cpp'
$auditedPin = '3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70'
$pin = (& git -C $repo rev-parse HEAD:llama.cpp).Trim()
if ($LASTEXITCODE -ne 0 -or $pin -ne $auditedPin) {
    throw "Re-audit the mtmd API/build before changing the llama.cpp pin (expected $auditedPin, got $pin)."
}
$actual = (& git -C $llama rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actual -ne $pin) {
    throw "llama.cpp checkout does not match the gitlink ($pin). Run git submodule update --init --recursive."
}

[xml]$app = Get-Content -Raw (Join-Path $repo 'uwp/xllama.vcxproj')
[xml]$lib = Get-Content -Raw (Join-Path $repo 'uwp/ggml-uwp.vcxproj')
function Nodes($document, $name) {
    @($document.SelectNodes("//*[local-name()='$name']"))
}
$backendCondition = "'`$(XllamaBackend)' == 'llamacpp' or '`$(XllamaBackend)' == 'unified'"
$define = @(Nodes $app 'XllamaMtmdDefine')
if ($define.Count -ne 1 -or $define[0].InnerText -ne 'XLLAMA_HAS_MTMD=1;' -or
    $define[0].GetAttribute('Condition') -ne $backendCondition) {
    throw 'The unified/llamacpp project must define XLLAMA_HAS_MTMD=1 unconditionally for those backends.'
}
$compilerDefs = (Nodes $app 'PreprocessorDefinitions' | ForEach-Object InnerText) -join ';'
if (-not $compilerDefs.Contains('$(XllamaMtmdDefine)')) {
    throw 'XllamaMtmdDefine is not passed to the app compiler.'
}
$vision = @(Nodes $app 'ClCompile' | Where-Object { $_.GetAttribute('Include') -eq 'vision_mtmd.cpp' })
if ($vision.Count -ne 1 -or $vision[0].GetAttribute('Condition') -or $vision[0].ExcludedFromBuild) {
    throw 'vision_mtmd.cpp must be compiled into the app (including its explicit ORT-only stub).'
}
$reference = @(Nodes $app 'ProjectReference' | Where-Object { $_.GetAttribute('Include') -eq 'ggml-uwp.vcxproj' })
if ($reference.Count -ne 1 -or $reference[0].ParentNode.GetAttribute('Condition') -ne $backendCondition -or
    $reference[0].LinkLibraryDependencies -eq 'false') {
    throw 'The unified/llamacpp app must link the static ggml/llama/mtmd project.'
}
foreach ($pair in @(
    @($app, '$(ProjectDir)..\llama.cpp\tools\mtmd;'),
    @($lib, '$(ProjectDir)..\llama.cpp\tools\mtmd;'),
    @($lib, '$(ProjectDir)..\llama.cpp\vendor;'),
    @($lib, '$(ProjectDir)..\llama.cpp;')
)) {
    $includes = (Nodes $pair[0] 'AdditionalIncludeDirectories' | ForEach-Object InnerText) -join ';'
    if (-not $includes.Contains($pair[1])) { throw "Missing mtmd include directory: $($pair[1])" }
}

# Compare exactly with upstream's library target, excluding CLI/debug binaries.
$cmake = Get-Content -Raw (Join-Path $llama 'tools/mtmd/CMakeLists.txt')
$target = [regex]::Match($cmake, '(?s)add_library\(mtmd\s+(.*?)\)')
if (-not $target.Success) { throw 'Cannot identify the pinned mtmd CMake library target.' }
$expected = @([regex]::Matches($target.Groups[1].Value, '[\w/-]+\.cpp') | ForEach-Object { $_.Value.Replace('/', '\') })
$compiled = @()
foreach ($item in (Nodes $lib 'ClCompile')) {
    $include = $item.GetAttribute('Include')
    if (-not $include.StartsWith('..\llama.cpp\tools\mtmd\')) { continue }
    if ($item.GetAttribute('Condition') -or $item.ParentNode.GetAttribute('Condition') -or $item.ExcludedFromBuild) {
        throw "mtmd source is conditionally excluded: $include"
    }
    $relative = $include.Substring('..\llama.cpp\tools\mtmd\'.Length)
    $files = @(Get-ChildItem -Path (Join-Path $repo "uwp/$include") -File)
    if ($files.Count -eq 0) { throw "Missing mtmd source: $include" }
    foreach ($file in $files) {
        $compiled += $(if ($relative.Contains('\')) { 'models\' + $file.Name } else { $file.Name })
    }
}
$delta = @(Compare-Object ($expected | Sort-Object -Unique) ($compiled | Sort-Object -Unique))
if ($delta.Count) {
    throw "ggml-uwp mtmd sources differ from the pinned CMake library: $($delta | Out-String)"
}
$libDefs = (Nodes $lib 'PreprocessorDefinitions' | ForEach-Object InnerText) -join ';'
if ($libDefs -match '\bMTMD_VIDEO\b') { throw 'MTMD_VIDEO must remain disabled: UWP cannot launch ffmpeg.' }
$helper = @(Nodes $lib 'ClCompile' | Where-Object { $_.GetAttribute('Include').EndsWith('\mtmd-helper.cpp') })
if ($helper.Count -ne 1 -or $helper[0].PreprocessorDefinitions -notmatch '\bMA_NO_RUNTIME_LINKING\b' -or
    $helper[0].PreprocessorDefinitions -notmatch '\bMA_NO_THREADING\b' -or
    $helper[0].PreprocessorDefinitions -notmatch '\bMA_NO_WIN32_FILEIO\b') {
    throw 'The mtmd memory decoder must exclude unused miniaudio desktop loading/thread helpers.'
}
Write-Host "UWP vision inputs verified: llama.cpp $pin, $($expected.Count) mtmd sources, vision bridge, macro, static link."
