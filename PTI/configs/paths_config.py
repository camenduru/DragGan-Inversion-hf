## Pretrained models paths
e4e = 'PTI/pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = '../PTI/pretrained_models/ffhq.pkl'
style_clip_pretrained_mappers = ''
ir_se50 = 'PTI/pretrained_models/model_ir_se50.pth'
dlib = 'PTI/pretrained_models/align.dat'

## Dirs for output files
checkpoints_dir = 'PTI/checkpoints'
embedding_base_dir = 'PTI/embeddings'
styleclip_output_dir = 'PTI/StyleCLIP_results'
experiments_output_dir = 'PTI/output'

## Input info
### Input dir, where the images reside
input_data_path = ''
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'barcelona'

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = 'PTI/editings/interfacegan_directions/age.pt'
interfacegan_smile = 'PTI/editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'PTI/editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'PTI/editings/ganspace_pca/ffhq_pca.pt'
