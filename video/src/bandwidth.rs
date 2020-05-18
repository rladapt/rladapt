extern crate cv;
extern crate darknet;
extern crate env_logger;
extern crate time;
extern crate video_analytics;
use cv::cuda::GpuHog as Hog;
use cv::objdetect::{HogParams, ObjectDetect, SvmDetector};
use darknet::*;
use std::env;
use std::fs::File;
use std::io::Write;

use video_analytics::loader::*;

fn main() {
    env_logger::init().unwrap();

    let skip = env::var("SKIP")
        .unwrap_or("0".to_string())
        .parse::<usize>()
        .expect("invalid SKIP via environment variable");

    let width = env::var("WIDTH")
        .unwrap_or("1920".to_string())
        .parse::<usize>()
        .expect("invalid WIDTH via environment variable");

    let quantizer = env::var("Q")
        .unwrap_or("20".to_string())
        .parse::<usize>()
        .expect("invalid Q via environment variable");

    let fname = env::var("FILE")
        .unwrap_or("output".to_string())
        .parse::<String>()
        .expect("invalid FILE via environment variable");

    let height = width / 16 * 9;
    
    let path = env::var("INPUT").expect("please specify the path for input images");
    let ext = env::var("EXT").expect("please specify the extension for input images");

    let lc = LoaderConfig {
        path: path,
        ext: ext,
        circular: false,
    };

    let config = VideoConfig {
        width: width,
        height: height,
        skip: skip,
        quantizer: quantizer,
    };
    let (loader, _loader_ctl) = load_x264(lc, config).unwrap();

    let mut i = 1;
    let mut sink_file = File::create(&format!("{}", fname)).unwrap();
    loop {
        // println!("{} ms", elapsed.subsec_nanos() / 1_000_000);
        let encoded = loader.recv().expect("failed to receive encoded");
        sink_file
            .write(&encoded)
            .expect("failed to write to file sink");
        println!("{}, {}", i, encoded.len());
        i += 1;
    }
}