use onnxruntime::{environment::Environment, session::InferenceSession, tensor::OrtOwnedTensor};
use opencv::{
    core::{self, Mat, Scalar, Size},
    highgui, imgproc,
    prelude::*,
};

fn main() -> anyhow::Result<()> {
    // Load image using OpenCV
    let img = Mat::from_path(
        "/Users/muntakim/Personal_coding/torch_yolo_expr/src/test.jpg",
        core::IMREAD_COLOR,
    )?;

    // Resize image to the size expected by the model
    let size = Size::new(416, 416);
    let mut resized = Mat::default();
    imgproc::resize(&img, &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

    // Convert the image to the format expected by the model (RGB in this case)
    let mut input = Mat::default();
    imgproc::cvt_color(&resized, &mut input, imgproc::COLOR_BGR2RGB)?;

    // Load the ONNX model using the `onnxruntime` crate
    let env = Environment::builder().build()?;
    let model_path = "/Users/muntakim/Personal_coding/torch_yolo_expr/src";
    let session = InferenceSession::new(&env, &model_path)?;

    // Prepare the input tensor for the model
    let mut input_data = vec![0u8; (size.width * size.height * 3) as usize];
    input.copy_to_slice(&mut input_data);
    let input_shape = [1, 3, size.height as i64, size.width as i64];
    let input_tensor = OrtOwnedTensor::from_shape_vec_and_dim(&input_shape, input_data)?;

    // Run inference on the input tensor using the ONNX model
    let output_tensors = session.run(vec![("input_1", &input_tensor)])?;

    // Process the output tensor returned by the model to obtain the object detection results
    let output_tensor = &output_tensors[0];
    let output_data = output_tensor.as_slice::<f32>()?;
    let num_classes = 80;
    let num_boxes = output_tensor.shape()[1] as usize;
    let mut results = vec![];
    for i in 0..num_boxes {
        let offset = i * (num_classes + 5);
        let x = output_data[offset];
        let y = output_data[offset + 1];
        let width = output_data[offset + 2];
        let height = output_data[offset + 3];
        let confidence = output_data[offset + 4];
        let mut class_id = 0;
        let mut class_score = output_data[offset + 5];
        for j in 1..num_classes {
            let score = output_data[offset + 5 + j];
            if score > class_score {
                class_id = j;
                class_score = score;
            }
        }
        if confidence * class_score > 0.5 {
            let left = (x - width / 2.0) * img.cols() as f32;
            let top = (y - height / 2.0) * img.rows() as f32;
            let right = (x + width / 2.0) * img.cols() as f32;
            let bottom = (y + height / 2.0) * img.rows() as f32;
            let class_name = format!("class {}", class_id);
            let result = (
                class_name,
                confidence * class_score,
                left,
                top,
                right,
                bottom,
            );
            results.push(result);
        }
    }
    for result in results {
        let (class_name, _, left, top, right, bottom) = result;
        let color = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let thickness = 2;
        let font = imgproc::FONT_HERSHEY_SIMPLEX;
        let font_scale = 0.5;
        let text_size = imgproc::get_text_size(&class_name, font, font_scale, thickness, None)?;
        let text_origin = core::Point::new(left as i32, top as i32 - text_size.height);
        imgproc::rectangle(
            &mut img,
            core::Rect::new(
                left as i32,
                top as i32,
                (right - left) as i32,
                (bottom - top) as i32,
            ),
            color,
            thickness,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::put_text(
            &mut img,
            &class_name,
            text_origin,
            font,
            font_scale,
            color,
            thickness,
            imgproc::LINE_8,
            false,
        )?;
    }

    // Display the result
    highgui::imshow("Result", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}
