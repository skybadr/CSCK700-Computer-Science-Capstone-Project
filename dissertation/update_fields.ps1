# Open the built dissertation in Word, refresh every field (TOC, lists of
# tables/figures, caption numbers), save, and export a PDF for checking.
$ErrorActionPreference = "Stop"
$docx = "C:\Capstone Project\dissertation\build\Abouabdou_Badr_Dissertation.docx"
$pdf  = "C:\Capstone Project\dissertation\build\Abouabdou_Badr_Dissertation.pdf"
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
