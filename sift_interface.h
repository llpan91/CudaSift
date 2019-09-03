// Copyright 2019 tuSimple. All Rights Reserved.
// Author: Liangliang Pan (liangliang.pan@tusimple.ai)

// TODO list
// add epolo constrain



#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

#pragma once

using namespace cv;

typedef std::vector<std::pair<cv::Point2f, cv::Point2f> > Matches;

class SiftInterface {
 public:
  SiftInterface(cv::Mat k_mat, unsigned int width, unsigned int height, int num_octaves = 3,
		float init_bulr = 1.0, float thresh = 3.5, float min_scale = 0.0, 
		bool up_scale = false);

  void extractFeature(cv::Mat& img, SiftData& sift_data);

  void extractMoreFeature(cv::Mat &img, SiftData &sift_data);
  
  void extractSubImg(cv::Mat &img, SiftData &sift_data);
  
  void uniformFeature(cv::Mat &img , SiftData& sift_data);
  
  void twoStageUniformFeature(cv::Mat& img, SiftData& sift_data);
  
  void extractFeatureBucket(const cv::Mat& image, SiftData& sift_data);
  
  std::vector<cv::KeyPoint> getKeyPoints(SiftData& sift_data);

  Matches getCoarseMatches(SiftData& sift_data1, SiftData& sift_data2);
  
  Matches rejectWithF(const Matches &matches);
  
  double fundmentalError(const cv::Mat f_mat, cv::Point2f pt1, cv::Point2f pt2);
  
  Matches getMatchesByHom(SiftData &data, SiftData &data2, float thresh);
  
  Matches matchDoubleCheck(SiftData &sift_data1, SiftData &sift_data2);
  
 private:
  cv::Mat k_mat_;

  cv::Mat mask_;
  
  int MIN_DIST_ = 10;
  
  // just initial once
  CudaImage img_model_;
  // SiftData sift_data1_, sift_data2_;
  unsigned int width_, height_;
  
  CudaImage sub_img_model_;
  unsigned int sub_width_, sub_height_;


  int num_octaves_ = 3;    /* Number of octaves in Gaussian pyramid */
  float init_blur_ = 1.0f; /* Amount of initial Gaussian blurring in standard deviations */
  float thresh_ = 2.0f;	   /* Threshold on difference of Gaussians for feature pruning */
  float min_scale_ = 0.0f; /* Minimum acceptable scale to remove fine-scale features */
  bool up_scale_ = false;  /* Whether to upscale image before extraction */
  
  // for feature extract by bucket
  int max_feature_num_per_subregion_ = 20;

};
