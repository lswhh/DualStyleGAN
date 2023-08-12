import os
import numpy as np
import torch
from util import save_image, load_image
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp

global g_ema, psp, transform
is_initialized = False

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Exemplar-Based Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data/content/081680.jpg', help="path of the content image")
        self.parser.add_argument("--style", type=str, default='cartoon', help="target style type")
        self.parser.add_argument("--style_id", type=int, default=53, help="the id of the style image")
        self.parser.add_argument("--truncation", type=float, default=0.75, help="truncation for intrinsic style code (content)")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[0.75]*7+[1]*11, help="weight of the extrinsic style")
        self.parser.add_argument("--name", type=str, default='cartoon_transfer', help="filename to save the generated images")
        self.parser.add_argument("--preserve_color", action="store_true", help="preserve the color of the content image")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator.pt', help="name of the saved dualstylegan")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--align_face", action="store_true", help="apply face alignment to the content image")
        self.parser.add_argument("--exstyle_name", type=str, default=None, help="name of the extrinsic style codes")
        self.parser.add_argument("--wplus", action="store_true", help="use original pSp encoder to extract the intrinsic style code")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_name is None:
            if os.path.exists(os.path.join(self.opt.model_path, self.opt.style, 'refined_exstyle_code.npy')):
                self.opt.exstyle_name = 'refined_exstyle_code.npy'
            else:
                self.opt.exstyle_name = 'exstyle_code.npy'        
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
def run_alignment(args):
    import dlib
    from model.encoder.align_all_parallel import align_face
    modelname = os.path.join(args.model_path, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    predictor = dlib.shape_predictor(modelname)
    aligned_image = align_face(filepath=args.content, predictor=predictor)
    return aligned_image


if __name__ == "__main__":
    #device = "cuda"
    device = "cpu"

    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    generator.eval()

    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)
    
    if args.wplus:
        model_path = os.path.join(args.model_path, 'encoder_wplus.pt')
    else:
        model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    if 'output_size' not in opts:
        opts['output_size'] = 1024    
    opts = Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(device)

    exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle='TRUE').item()

    z_plus_latent=not args.wplus
    return_z_plus_latent=not args.wplus
    input_is_latent=args.wplus    
    
    print('Load models successfully!')
    
    with torch.no_grad():
        viz = []
        # load content image
        if args.align_face:
            I = transform(run_alignment(args)).unsqueeze(dim=0).to(device)
            I = F.adaptive_avg_pool2d(I, 1024)
        else:
            I = load_image(args.content).to(device)
        viz += [I]

        # reconstructed content image and its intrinsic style code
        img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                                   z_plus_latent=z_plus_latent, return_z_plus_latent=return_z_plus_latent, resize=False)  
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        viz += [img_rec]

        stylename = list(exstyles.keys())[args.style_id]
        latent = torch.tensor(exstyles[stylename]).to(device)
        if args.preserve_color and not args.wplus:
            latent[:,7:18] = instyle[:,7:18]
        # extrinsic styte code
        exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)
        if args.preserve_color and args.wplus:
            exstyle[:,7:18] = instyle[:,7:18]
            
        # load style image if it exists
        S = None
        if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
            S = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename)).to(device)
            viz += [S]

        # style transfer 
        # input_is_latent: instyle is not in W space
        # z_plus_latent: instyle is in Z+ space
        # use_res: use extrinsic style path, or the style is not transferred
        # interp_weights: weight vector for style combination of two paths
        img_gen, _ = generator([instyle], exstyle, input_is_latent=input_is_latent, z_plus_latent=z_plus_latent,
                              truncation=args.truncation, truncation_latent=0, use_res=True, interp_weights=args.weight)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        viz += [img_gen]

    print('Generate images successfully!')
    
    save_name = args.name+'_%d_%s'%(args.style_id, os.path.basename(args.content).split('.')[0])
    save_image(torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256), 4, 2).cpu(), 
               os.path.join(args.output_path, save_name+'_overview.jpg'))
    save_image(img_gen[0].cpu(), os.path.join(args.output_path, save_name+'.jpg'))

    print('Save images successfully!')




def init_cartoonize(model_path, generator_name, encoder_name):
    global g_ema, psp, transform, is_initialized

    if is_initialized:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # generator 모델 로드
    gen_ckpt = torch.load(os.path.join(model_path, generator_name), map_location=device)
    if "g_ema" in gen_ckpt:
        g_ema = gen_ckpt["g_ema"]
    else:
        g_ema = DualStyleGAN.from_layers(gen_ckpt[0], gen_ckpt[1], gen_ckpt[2], gen_ckpt[3], gen_ckpt[4])

    g_ema.to(device).eval()

    # encoder 모델 로드
    psp = pSp(encoder_torch_path=os.path.join(model_path, encoder_name)).to(device).eval()

    # 이미지 변환 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 초기화 상태 변경
    is_initialized = True
    
def cartoonize(content_image: Image.Image) -> Image.Image:
    global g_ema, psp, transform

    if not is_initialized:
        init_cartoonize(model_path="./checkpoint/",
                        generator_name="generator.pt",
                        encoder_name="psp.pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = Namespace(
        content=content_image,
        style="cartoon",
        style_id=53,
        truncation=0.75,
        weight=[0.75] * 7 + [1] * 11,
        preserve_color=True,
        model_path="./checkpoint/",
        model_name="generator.pt",
        data_path="./data/",
        wplus=False,
    )
    content_image_tensor = transform(content_image).unsqueeze(0).to(device)

    ckpt = torch.load(os.path.join(args.model_path, args.model_name), map_location=device)
    if "g_ema" in ckpt:
        g_ema = ckpt["g_ema"]
    else:
        g_ema = DualStyleGAN.from_layers(ckpt[0], ckpt[1], ckpt[2], ckpt[3], ckpt[4])
    g_ema.to(device).eval()
    
    psp = pSp(encoder_torch_path=os.path.join(args.model_path, args.encoder_name)).to(device).eval()

    instyle = torch.randn(content_image_tensor.size(0), g_ema.style_dim).to(device)
    residual = (content_image_tensor - g_ema.style_space).sum(2).sum(2) / (content_image_tensor.size(2) * content_image_tensor.size(3))
    instyle[:, :7] = residual.detach()

    with torch.no_grad():
        z_inter = psp(content_image_tensor)
        exstyle, _ = g_ema.mapping(z_inter, None, truncation_cutoff=-1)

    if args.preserve_color and not args.wplus:
        exstyle[:,7:18] = instyle[:,7:18]

    exstyle = exstyle.cpu().numpy()
    scaleY = exstyle[:, 7:18]
    scaleY = scaleY ** 2
    scaleY = scaleY + 1
    scaleY = scaleY / 2
    scaleY = scaleY * 255
    scaleY = scaleY.round()
    scaleY = scaleY.astype(int)
    for i, m in enumerate(scaleY):
        scaleX = np.asarray([53] * 11)
        _, x = torchmeshgrid(torch.as_tensor(x).float(), torch.as_tensor(m).float(), device)
        x = x / 63
        x = 2 * x - 1
        x = torch.as_tensor(x).float().to(device)
        exstyle[i, 7:18] = x

    exstyle = torch.tensor(exstyle).float().to(device)

    img_gen = g_ema.generate(content_image_tensor, exstyle, step=options.step, alpha=options.alpha,
                                  noise=None, mix_styles=options.wplus)
    
    img_gen_pil = Image.fromarray(np.uint8((img_gen.squeeze().numpy() + 1) / 2 * 255)).convert("RGB")
    return img_gen_pil