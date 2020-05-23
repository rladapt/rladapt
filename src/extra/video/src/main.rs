extern crate cv;
extern crate darknet;
extern crate env_logger;
extern crate time;
extern crate video_analytics;
extern crate rand;
use cv::cuda::GpuHog as Hog;
use cv::objdetect::{HogParams, ObjectDetect, SvmDetector};
use darknet::*;
use std::env;
use std::fs::File;
use std::io::Write;
use rand::{thread_rng, Rng};

use video_analytics::loader::*;

fn main() {
    env_logger::init().unwrap();
    pedestrian();
}

fn pedestrian() {
    let path = env::var("INPUT").expect("please specify the path for input video ");
    let nums = env::var("NUMS").expect("please specify the path for input num");
    let skip = env::var("SKIP").expect("please specify the skip num");
    // let cap = cv::videoio::VideoCapture::from_path(&path);

    // Prepare HOG detector
    let mut params = HogParams::default();
    params.hit_threshold = 0.3;
    let mut hog = Hog::with_params(params);
    let detector = SvmDetector::default_people_detector();
    hog.set_svm_detector(detector);

    let mut frame_no = 0;

    // let mut vec:Vec<u32> = (1..500).collect();
    // thread_rng().shuffle(&mut vec);
    let vec: Vec<&str> = nums.split(',').collect();
    let skip = skip.parse::<i32>().unwrap();
    for i in &vec {
        // while let Some(image) = cap.read() {
        // let image = image.cvt_color(cv::imgproc::ColorConversionCodes::BGR2RGB);
        let f = format!("{}/{:06}.jpg", path, i.parse::<i32>().unwrap());
        println!("new_frame\n{}", f);
        if skip == 0 || frame_no % skip == 0 {
            let image = cv::Mat::from_path(&f, cv::imgcodecs::ImreadModes::ImreadGrayscale).unwrap();
            //    while let Some(image) = cap.read() {
            //        let image = image.cvt_color(cv::imgproc::ColorConversionCodes::BGR2GRAY);
            let time = ::std::time::Instant::now();
            // Result is a vector of tuple (Rect, conf: f64). See documentation
            // of hog detection if you are confused.
            let result = hog.detect(&image);
            let elapsed = time.elapsed();
            let proc_time =
                elapsed.as_secs() as f64 * 1_000.0 + elapsed.subsec_nanos() as f64 / 1_000_000.0;

            for r in &result {
                let normalized = r.0.normalize_to_mat(&image);
                println!(
                    "{:06},{:.02},{},{},{},{},{},{}",
                    i.parse::<i32>().unwrap(),
                    proc_time,
                    "pedestrian",
                    r.1,
                    normalized.x,
                    normalized.y,
                    normalized.width,
                    normalized.height
                );
            }

            if result.len() == 0{
                println!(
                    "{:06},{:.02},{},{},{},{},{},{}",
                    i.parse::<i32>().unwrap(),
                    proc_time,
                    "None",
                    0,
                    0,
                    0,
                    0,
                    0
                );
            }
        } else {
            println!(
                "{:06},{:.02},{},{},{},{},{},{}",
                i.parse::<i32>().unwrap(),
                0,
                "None",
                0,
                0,
                0,
                0,
                0
            );
        }

        frame_no += 1
    }
}
