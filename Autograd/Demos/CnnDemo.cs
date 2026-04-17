using Autograd.Engine.Core;
using Autograd.Engine.Enums;

namespace Autograd.Demos;

public class CnnDemo : IDemo
{
    private const int Epochs = 1000;
    private const int SamplesPerEpoch = 32;
    private const int ImageSize = 32;
    private const float LearningRate = 0.001f;
    private const int DecayHalfLifeEpochs = 200;

    public string Name => "Convolutional Neural Network";

    public void Run()
    {
        // Three 3x3 conv layers: 16->14->12->10
        CNN.CNN cnn = CNN.CNN.Create(1)
                             .WithLayer(3, ActivationType.ReLU)
                             .WithLayer(3, ActivationType.ReLU)
                             .WithOutput(3);

        Random random = new(42);
        float[] lossHistory = new float[Epochs];

        for (int epoch = 0; epoch < Epochs; epoch++)
        {
            float epochLoss = 0;
            float lr = LearningRate; // todo: decay ?

            for (int s = 0; s < SamplesPerEpoch; s++)
            {
                float[] image = GenerateImage(random);
                float[] sobel = ComputeSobel(image, ImageSize, ImageSize);
                float[] gt = CropCenter(sobel, ImageSize - 2, ImageSize - 6);

                Tensor input = new(image, [1, 1, ImageSize, ImageSize]);
                Tensor groundTruth = new(gt, [1, 1, ImageSize - 6, ImageSize - 6]);

                Tensor output = cnn.Forward(input);
                Tensor mse = Tensor.MSE(output, groundTruth);

                epochLoss += mse.GetData()[0];

                mse.Backward();
                cnn.Adjust(lr);
                cnn.Zero();
            }

            epochLoss /= SamplesPerEpoch;
            lossHistory[epoch] = epochLoss;
            Console.WriteLine($"EPOCH {epoch + 1}, LOSS: {epochLoss}, LR: {lr}");
        }

        Console.WriteLine($"TRAINING FINISHED. LOSS: {lossHistory[^1]}");

        // Export visualization data using a showcase image
        float[] showcaseImage = GenerateImage(new Random(100));
        Tensor showcaseInput = new(showcaseImage, [1, 1, ImageSize, ImageSize]);
        Tensor pred = cnn.Forward(showcaseInput);

        float[] sobelShowcase = ComputeSobel(showcaseImage, ImageSize, ImageSize);
        float[] gtShowcase = CropCenter(sobelShowcase, ImageSize - 2, ImageSize - 6);

        int outSize = ImageSize - 6;

        float[][] input2D = DumpHelper.ReshapeTo2D(showcaseImage, ImageSize, ImageSize);
        float[][] pred2D = DumpHelper.ReshapeTo2D(pred.GetData(), outSize, outSize);
        float[][] gt2D = DumpHelper.ReshapeTo2D(gtShowcase, outSize, outSize);

        DumpHelper.DumpCnnData("cnn_data.json", input2D, pred2D, gt2D, lossHistory);
    }

    /// <summary>
    /// Generate a synthetic grayscale image with random geometric shapes
    /// </summary>
    private static float[] GenerateImage(Random r)
    {
        float[] img = new float[ImageSize * ImageSize];

        // Draw 1-3 rectangles
        int rectCount = r.Next(1, 4);
        for (int n = 0; n < rectCount; n++)
        {
            int x1 = r.Next(0, ImageSize - 4);
            int y1 = r.Next(0, ImageSize - 4);
            int x2 = r.Next(x1 + 3, Math.Min(x1 + 18, ImageSize));
            int y2 = r.Next(y1 + 3, Math.Min(y1 + 18, ImageSize));
            float intensity = 0.5f + r.NextSingle() * 0.5f;

            for (int y = y1; y < y2; y++)
                for (int x = x1; x < x2; x++)
                    img[y * ImageSize + x] = intensity;
        }

        // Draw 1-2 circles
        int circleCount = r.Next(1, 3);
        for (int n = 0; n < circleCount; n++)
        {
            int cx = r.Next(5, ImageSize - 5);
            int cy = r.Next(5, ImageSize - 5);
            int radius = r.Next(3, 9);
            float intensity = 0.5f + r.NextSingle() * 0.5f;

            for (int y = 0; y < ImageSize; y++)
            {
                for (int x = 0; x < ImageSize; x++)
                {
                    float dist = MathF.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                    if (dist <= radius)
                        img[y * ImageSize + x] = intensity;
                }
            }
        }

        return img;
    }

    /// <summary>
    /// Apply Sobel edge detection. Output is (w-2)x(h-2).
    /// </summary>
    private static float[] ComputeSobel(float[] img, int w, int h)
    {
        int ow = w - 2;
        int oh = h - 2;
        float[] result = new float[ow * oh];
        float max = 0;

        for (int y = 0; y < oh; y++)
        {
            for (int x = 0; x < ow; x++)
            {
                // Sobel Gx kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]
                float gx = -img[(y) * w + x] + img[(y) * w + (x + 2)]
                          - 2 * img[(y + 1) * w + x] + 2 * img[(y + 1) * w + (x + 2)]
                          - img[(y + 2) * w + x] + img[(y + 2) * w + (x + 2)];

                // Sobel Gy kernel: [[-1,-2,-1],[0,0,0],[1,2,1]]
                float gy = -img[(y) * w + x] - 2 * img[(y) * w + (x + 1)] - img[(y) * w + (x + 2)]
                          + img[(y + 2) * w + x] + 2 * img[(y + 2) * w + (x + 1)] + img[(y + 2) * w + (x + 2)];

                float magnitude = MathF.Sqrt(gx * gx + gy * gy);
                result[y * ow + x] = magnitude;
                if (magnitude > max) max = magnitude;
            }
        }

        // Normalize to [0, 1]
        if (max > 0)
        {
            for (int i = 0; i < result.Length; i++)
                result[i] /= max;
        }

        return result;
    }

    /// <summary>
    /// Center-crop a square image from fromSize to toSize
    /// </summary>
    private static float[] CropCenter(float[] img, int fromSize, int toSize)
    {
        int offset = (fromSize - toSize) / 2;
        float[] result = new float[toSize * toSize];

        for (int y = 0; y < toSize; y++)
            for (int x = 0; x < toSize; x++)
                result[y * toSize + x] = img[(y + offset) * fromSize + (x + offset)];

        return result;
    }
}
