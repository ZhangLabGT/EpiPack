import numpy as np
import torch


def query_model_initial(query_model = None, ref_model_param = None, query_batch_num=None, gene_feature_num = 3000, latent_embedding_dim = 50):
    
    ref_para = ref_model_param
    device = ref_para['Encoder.fc.fc_layers.Layer 0.0.weight'].device

    query_para = ref_para

    #initial first encoder layer
    feature_weight = ref_para['Encoder.fc.fc_layers.Layer 0.0.weight'][:,:gene_feature_num]
    new_batch_weight = np.random.randn(feature_weight.shape[0], query_batch_num) / np.sqrt(feature_weight.shape[0])
    new_batch_weight = torch.from_numpy(new_batch_weight).float().to(device)
    init_weight_en = torch.cat([feature_weight, new_batch_weight], dim=1)
    ref_model_param['Encoder.fc.fc_layers.Layer 0.0.weight']

    query_para['Encoder.fc.fc_layers.Layer 0.0.weight'] = init_weight_en

    #initial first decoder layer
    latent_weight = ref_para['Decoder.fc.fc_layers.Layer 0.0.weight'][:,:latent_embedding_dim]
    new_batch_weight = np.random.randn(latent_weight.shape[0], query_batch_num) / np.sqrt(latent_weight.shape[0])
    new_batch_weight = torch.from_numpy(new_batch_weight).float().to(device)
    init_weight_de = torch.cat([latent_weight, new_batch_weight], dim=1)

    query_para['Decoder.fc.fc_layers.Layer 0.0.weight'] = init_weight_de

    query_model.load_state_dict(query_para)
    
    fix = ['Layer 1', 'mean_layer', 'var_layer']
    for k,v in query_model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in fix):
            v.requires_grad = False

    return query_model
    


def cosine_dist(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return 1 - ip / torch.ger(w1,w2).clamp(min=eps)

    