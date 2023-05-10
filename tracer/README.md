# TRACER
Image Masking approach


## Run **main.py**
<pre><code>
TRACER
├── data
│   ├── custom_dataset
│   │   ├── sample_image1.png
│   │   ├── sample_image2.png
      .
      .
      .

# For testing TRACER with pre-trained model (e.g.)  
python main.py inference --dataset custom_dataset/ --arch 7 --img_size 640 --save_map True
</code></pre>

## Requirements
* Python >= 3.7.x
* Pytorch >= 1.8.0
* albumentations >= 0.5.1
* tqdm >=4.54.0
* scikit-learn >= 0.23.2

## Configurations
--arch: EfficientNet backbone scale: TE0 to TE7.  
--frequency_radius: High-pass filter radius in the MEAM.  
--gamma: channel confidence ratio \gamma in the UAM.   
--denoise: Denoising ratio d in the OAM.  
--RFB_aggregated_channel: # of channels in receptive field blocks.  
--multi_gpu: Multi-GPU learning options.  
--img_size: Input image resolution.  
--save_map: Options saving predicted mask.  

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Img size</th>
  </tr>
</thead>
<tbody>
    <tr>
        <td>TRACER-Efficient-0 ~ 1</td>
        <td>320</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-2</td>
        <td>352</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-3</td>
        <td>384</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-4</td>
        <td>448</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-5</td>
        <td>512</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-6</td>
        <td>576</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-7</td>
        <td>640</td>
    </tr>
</tbody>
</table>
