# Open the built dissertation in Word, refresh every field (TOC, lists of
# tables/figures, caption numbers), save, and export a PDF for checking.
# Usage: update_fields.ps1 [-Name Abouabdou_Badr_Dissertation_v2]
param([string]$Name = "Abouabdou_Badr_Dissertation")
$ErrorActionPreference = "Stop"
$docx = "C:\Capstone Project\dissertation\build\$Name.docx"
$pdf  = "C:\Capstone Project\dissertation\build\$Name.pdf"

# refuse to run on a file that is open elsewhere (Word would hang on a hidden prompt)
try { $fs = [System.IO.File]::Open($docx, 'Open', 'ReadWrite', 'None'); $fs.Close() }
catch { Write-Output "LOCKED: $docx is open in another program; close it or use -Name"; exit 1 }

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx)
    for ($i = 0; $i -lt 2; $i++) {
        $doc.Fields.Update() | Out-Null
        foreach ($t in $doc.TablesOfContents) { $t.Update() }
        foreach ($t in $doc.TablesOfFigures) { $t.Update() }
    }
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)
    $pages = $doc.ComputeStatistics(2)
    $words = $doc.ComputeStatistics(0)
    Write-Output "pages=$pages words=$words"
    $doc.Close([ref]0)
} finally {
    $word.Quit()
}
