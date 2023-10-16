from PIL import Image
import torch.nn as nn
from lavis.models import load_model_and_preprocess

cosim = nn.CosineSimilarity(dim=1, eps=1e-08)
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

device = 'cuda'
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
caption = "a person holding a camera on a grass field during the sunset"
raw_image = Image.open('dataset/test.jpeg')
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
text_input = txt_processors["eval"](caption)
sample = {"image": image, "text_input": [text_input]}

features_image = model.extract_features(sample, mode="image")
features_text = model.extract_features(sample, mode="text")
print(features_image.image_embeds.shape)
# torch.Size([1, 197, 768])
print(features_text.text_embeds.shape)
# torch.Size([1, 12, 768])

# low-dimensional projected features
print(features_image.image_embeds_proj.shape)
# torch.Size([1, 197, 256])
print(features_text.text_embeds_proj.shape)
# torch.Size([1, 12, 256])


print(features_image.image_embeds[:,0,:].shape)
# torch.Size([1, 197, 768])
print(features_text.text_embeds[:,0,:].shape)
# torch.Size([1, 12, 768])

# low-dimensional projected features
print(features_image.image_embeds_proj[:,0,:].shape)
# torch.Size([1, 197, 256])
print(features_text.text_embeds_proj[:,0,:].shape)
# torch.Size([1, 12, 256])


similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
print(similarity)
similarity = features_image.image_embeds[:,0,:] @ features_text.text_embeds[:,0,:].t()
print(similarity)

print(cosim(features_image.image_embeds_proj[:,0,:], features_text.text_embeds_proj[:,0,:]))
print(cosim(features_image.image_embeds[:,0,:], features_text.text_embeds[:,0,:]))