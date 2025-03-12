# Get the network adapter index
$adapter = Get-NetAdapter | Where-Object {$_.Status -eq "Up"} | Select-Object -First 1

# Backup current DNS settings
$currentDNS = Get-DnsClientServerAddress -InterfaceIndex $adapter.ifIndex -AddressFamily IPv4

# Set Google DNS servers
Set-DnsClientServerAddress -InterfaceIndex $adapter.ifIndex -ServerAddresses "8.8.8.8","8.8.4.4"

Write-Host "DNS servers set to Google DNS. Press Enter to restore original settings..."
$null = Read-Host

# Restore original DNS settings
Set-DnsClientServerAddress -InterfaceIndex $adapter.ifIndex -ServerAddresses $currentDNS.ServerAddresses 