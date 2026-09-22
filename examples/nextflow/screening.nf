/*
 * Proteus on GA4GH TES from Nextflow (nf-ga4gh plugin).
 *
 * Every process runs the `proteus` binary inside the proteus container on a TES server:
 *   MUTATE   — in-silico alanine scan of the scaffold sequence (one task)
 *   ANALYZE  — all-atom biophysics of each input structure (one task per file)
 *   SUMMARY  — gather the per-structure reports
 *
 * Folding is not part of this workflow: it is a scatter/gather of offline proteus commands, so it
 * runs unchanged on any TES 1.1 endpoint. Point `params.structures` at predicted models to score
 * them the same way.
 */

params.scaffold   = "${projectDir}/1crn.fasta"
params.structures = "${projectDir}/../../crates/proteus-core/tests/data/1crn.{pdb,cif}"
params.image      = "ghcr.io/otoyuki/proteus:latest"
params.outdir     = "${projectDir}/results"

process MUTATE {
    tag "${scaffold.baseName}"
    container params.image
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path scaffold

    output:
    path "variants.fasta"

    script:
    """
    set -euo pipefail
    proteus mutate ${scaffold} --mode alanine --start 1 --end 6 --output variants.fasta
    """
}

process ANALYZE {
    tag "${structure.name}"
    container params.image
    publishDir "${params.outdir}/metrics", mode: 'copy'

    input:
    path structure

    output:
    path "${structure.name}.metrics.txt"

    script:
    """
    set -euo pipefail
    proteus analyze --pdb ${structure} | tee ${structure.name}.metrics.txt
    """
}

process SUMMARY {
    container params.image
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path reports

    output:
    path "summary.txt"

    script:
    """
    set -euo pipefail
    {
        echo "structures analysed: ${reports.size()}"
        for f in ${reports}; do
            echo "== \$f"
            cat "\$f"
        done
    } > summary.txt
    """
}

workflow {
    MUTATE(Channel.fromPath(params.scaffold))
    ANALYZE(Channel.fromPath(params.structures, checkIfExists: true))
    SUMMARY(ANALYZE.out.collect())
}
