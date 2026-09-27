#!/usr/bin/env nextflow
// Binder triage with the PROTEUS_ANALYZE module: one Parquet table per design campaign.
//
//   nextflow run examples/nextflow/binder-triage \
//       --campaigns 'boltz_results/*' --interface A:B
//
// Each directory matched by --campaigns is one campaign; its structure files and the predictor's
// PAE/scores files beside them are staged together, so ipSAE, ipAE, LIS and ipTM are filled in.

include { PROTEUS_ANALYZE } from '../../../packaging/nf-core/modules/proteus/analyze/main'

workflow {
    if (!params.campaigns) {
        error "give --campaigns 'dir/*' (one directory per campaign)"
    }
    campaigns = channel
        .fromPath(params.campaigns, type: 'dir')
        .map { dir ->
            def files = []
            dir.eachFileRecurse(groovy.io.FileType.FILES) { f ->
                // AlphaFold 3's per-sample directories repeat the same file names; the top-ranked
                // model and its confidences sit one level up.
                if (!f.toString().contains('/seed-') &&
                    f.name =~ /\.(pdb|cif|mmcif|ent)(\.gz)?$|\.npz$|confidence.*\.json$|_scores_.*\.json$|_full_data_.*\.json$/) {
                    files << f
                }
            }
            [[id: dir.name], files]
        }
    PROTEUS_ANALYZE(campaigns)
    PROTEUS_ANALYZE.out.qc.view { meta, table -> "${meta.id}: ${table}" }
}
