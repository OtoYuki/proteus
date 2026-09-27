process PROTEUS_ANALYZE {
    tag "${meta.id}"
    label 'process_medium'

    conda "${moduleDir}/environment.yml"
    // Until the Bioconda package (and so a BioContainers image) exists, the release image.
    container "ghcr.io/otoyuki/proteus:0.9.0"

    input:
    // Structure files (.pdb/.cif, optionally .gz) together with whatever the predictor wrote
    // beside them (Boltz pae_*.npz and confidence_*.json, AlphaFold 3 *_confidences.json, …),
    // so the PAE-derived interface metrics can find them.
    tuple val(meta), path(models, stageAs: 'models/*')

    output:
    tuple val(meta), path("${prefix}.parquet"), emit: qc
    tuple val("${task.process}"), val('proteus'), eval("proteus --version | sed 's/^proteus //'"), topic: versions, emit: versions_proteus

    when:
    task.ext.when == null || task.ext.when

    script:
    // e.g. ext.args = '--interface A:B' for binder triage
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    proteus analyze models/ \\
        --export ${prefix}.parquet \\
        --jobs ${task.cpus} \\
        --top 0 \\
        ${args}
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.parquet
    """
}
