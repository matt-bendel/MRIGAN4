from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.fid.fid_metric import FIDMetric

def get_fid(args, G, ref_loader, cond_loader):
    print("GETTING VGG EMBEDDING")
    vgg_embedding = VGG16Embedding(parallel=True)

    print("GETTING DATA LOADERS")

    fid_metric = FIDMetric(gan=G,
                           ref_loader=ref_loader,
                           loader=cond_loader,
                           image_embedding=vgg_embedding,
                           condition_embedding=vgg_embedding,
                           cuda=True,
                           args=args)

    fid_metric.get_fid()