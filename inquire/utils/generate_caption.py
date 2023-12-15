from transformers import AutoProcessor, Blip2ForConditionalGeneration

import torch

# Use blip to generate caption
def generate_caption(image, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    if model is None:
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text

def main():
    import requests

    from PIL import Image
    from torchvision import transforms
    
    transform = transforms.Compose([transforms.PILToTensor()])

    # Get the image from the camera
    image = Image.open(requests.get('https://media.newyorker.com/cartoons/63dc6847be24a6a76d90eb99/master/w_1160,c_limit/230213_a26611_838.jpg', stream=True).raw).convert('RGB')
    image = transform(image)
    image = torch.stack([image]*3)
    print(image.shape)
    # Generate the caption
    caption = generate_caption(image)
    print(caption)
    

if __name__ == '__main__':
    main()