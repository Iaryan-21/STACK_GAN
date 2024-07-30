import torch
import unittest
from model import stageI_Gen, stageI_Dis, StageII_GeN, StageII_Disc

class TestGANNetwork(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.text_dim = 1024
        self.z_dim = 100
        self.img_size = 64  # Assuming 64x64 for Stage I
        self.img_size_stage2 = 256  # Assuming 256x256 for Stage II

    def test_stageI_Gen(self):
        model = stageI_Gen()
        text_embedding = torch.randn(self.batch_size, self.text_dim)
        noise = torch.randn(self.batch_size, self.z_dim)
        
        _, fake_img, mu, logvar = model(text_embedding, noise)
        
        self.assertEqual(fake_img.shape, (self.batch_size, 3, self.img_size, self.img_size))
        self.assertEqual(mu.shape, (self.batch_size, 128))
        self.assertEqual(logvar.shape, (self.batch_size, 128))

    def test_stageI_Dis(self):
        model = stageI_Dis()
        fake_img = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        
        img_embedding = model(fake_img)
        
        self.assertEqual(img_embedding.shape, (self.batch_size, model.df_dim * 8, self.img_size // 16, self.img_size // 16))

    def test_StageII_GeN(self):
        stageI_model = stageI_Gen()
        model = StageII_GeN(stageI_model)
        text_embedding = torch.randn(self.batch_size, self.text_dim)
        noise = torch.randn(self.batch_size, self.z_dim)
        
        stage1_img, fake_img, mu, logvar = model(text_embedding, noise)
        
        self.assertEqual(stage1_img.shape, (self.batch_size, 3, self.img_size, self.img_size))
        self.assertEqual(fake_img.shape, (self.batch_size, 3, self.img_size_stage2, self.img_size_stage2))
        self.assertEqual(mu.shape, (self.batch_size, 128))
        self.assertEqual(logvar.shape, (self.batch_size, 128))

    def test_StageII_Disc(self):
        model = StageII_Disc()
        fake_img = torch.randn(self.batch_size, 3, self.img_size_stage2, self.img_size_stage2)
        
        img_embedding = model(fake_img)
        
        expected_size = self.img_size_stage2 // 64  # Depends on the number of downsampling operations
        self.assertEqual(img_embedding.shape, (self.batch_size, model.df_dim * 8, expected_size, expected_size))

    def test_gradient_flow(self):
        model = stageI_Gen()
        text_embedding = torch.randn(self.batch_size, self.text_dim, requires_grad=True)
        noise = torch.randn(self.batch_size, self.z_dim, requires_grad=True)
        
        _, fake_img, _, _ = model(text_embedding, noise)
        loss = fake_img.mean()
        loss.backward()
        
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertNotEqual(torch.sum(param.grad).item(), 0, f"Zero gradient for {name}")

if __name__ == '__main__':
    unittest.main()