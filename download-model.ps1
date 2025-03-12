# Set TLS 1.2 and force IPv4
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
[System.Net.ServicePointManager]::DnsRefreshTimeout = 0
[System.Net.ServicePointManager]::UseNagleAlgorithm = $false

# Create models directory if it doesn't exist
$modelsDir = "$env:USERPROFILE\.ollama\models"
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force
}

# Use direct IP address for huggingface.co
$modelUrl = "https://54.192.171.56/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf"
$outputFile = "$modelsDir\mistral.gguf"

Write-Host "Attempting to resume Mistral model download..."

# Function to convert bytes to GB
function Convert-BytesToGB {
    param([long]$bytes)
    return [math]::Round($bytes/1GB, 2)
}

# Get current file size
$startPosition = if (Test-Path $outputFile) { (Get-Item $outputFile).Length } else { 0 }
Write-Host "Starting from position: $(Convert-BytesToGB $startPosition) GB"

# Maximum number of retries
$maxRetries = 5
$currentTry = 0
$success = $false

# Ignore SSL certificate errors (since we're using IP directly)
[System.Net.ServicePointManager]::ServerCertificateValidationCallback = {$true}

while (-not $success -and $currentTry -lt $maxRetries) {
    $currentTry++
    Write-Host "Attempt $currentTry of $maxRetries"
    
    try {
        $request = [System.Net.HttpWebRequest]::Create($modelUrl)
        $request.Timeout = 30 * 60 * 1000  # 30 minutes
        $request.ReadWriteTimeout = 30 * 60 * 1000  # 30 minutes
        $request.KeepAlive = $false
        $request.Host = "huggingface.co"
        
        if ($startPosition -gt 0) {
            $request.AddRange($startPosition)
        }
        
        Write-Host "Connecting to server..."
        $response = $request.GetResponse()
        $totalLength = $response.ContentLength + $startPosition
        $responseStream = $response.GetResponseStream()
        $fileStream = [System.IO.File]::OpenWrite($outputFile)
        
        if ($startPosition -gt 0) {
            $fileStream.Seek($startPosition, [System.IO.SeekOrigin]::Begin) | Out-Null
        }
        
        Write-Host "Total size: $(Convert-BytesToGB $totalLength) GB"
        $buffer = New-Object byte[] 1MB
        $count = 0
        $totalRead = $startPosition
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        
        do {
            $count = $responseStream.Read($buffer, 0, $buffer.Length)
            if ($count -gt 0) {
                $fileStream.Write($buffer, 0, $count)
                $totalRead += $count
                
                # Calculate progress and speed
                $percent = [math]::Round(($totalRead * 100) / $totalLength, 2)
                $speed = [math]::Round(($totalRead - $startPosition) / 1MB / $sw.Elapsed.TotalSeconds, 2)
                
                # Update progress
                Write-Progress -Activity "Downloading Mistral model" `
                    -Status "Downloaded: $(Convert-BytesToGB $totalRead) GB of $(Convert-BytesToGB $totalLength) GB ($speed MB/s)" `
                    -PercentComplete $percent
            }
        } while ($count -gt 0)
        
        Write-Host "Download complete!"
        $success = $true
        
    } catch {
        Write-Host "Error during attempt $currentTry`: $_"
        if ($currentTry -lt $maxRetries) {
            Write-Host "Waiting 10 seconds before next attempt..."
            Start-Sleep -Seconds 10
        }
    } finally {
        if ($fileStream) { $fileStream.Close() }
        if ($responseStream) { $responseStream.Close() }
        if ($response) { $response.Close() }
    }
}

if ($success) {
    $finalSize = (Get-Item $outputFile).Length
    Write-Host "Final file size: $(Convert-BytesToGB $finalSize) GB"
    Write-Host "Now try running: ollama create mistral-local -f $outputFile"
} else {
    Write-Host "Failed to download after $maxRetries attempts."
    Write-Host "Please try these alternatives:"
    Write-Host "1. Try downloading directly in your browser: https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf"
    Write-Host "2. Try a smaller model version: mistral-7b-v0.1.Q3_K_S.gguf"
    Write-Host "3. Check your internet connection and try again later"
} 