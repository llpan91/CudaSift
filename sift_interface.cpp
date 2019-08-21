// Copyright 2019 tuSimple. All Rights Reserved.
// Author: Liangliang Pan (liangliang.pan@tusimple.ai)

#include <Eigen/Core>
#include "sift_interface.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

SiftInterface::SiftInterface(cv::Mat k_mat, unsigned int width, unsigned int height,
                             int num_octaves, float init_bulr, float thresh, float min_scale, 
			     bool up_scale) {
  k_mat_ = k_mat.clone();
  width_ = width;
  height_ = height;
  
  num_octaves_ = num_octaves;
  init_blur_ = init_bulr;
  thresh_ = thresh;
  min_scale_ = min_scale;
  up_scale_ = up_scale;
  
  InitCuda(0);
  img_model_.setParam(width_, height_, iAlignUp(width_, 128), false, NULL);
}

void SiftInterface::extractFeature(cv::Mat &img, SiftData &sift_data) {
  cv::Mat img_tmp;
  img.convertTo(img_tmp, CV_32FC1);

  img_model_.setImageIntiGPU((float *)img_tmp.data);
  img_model_.Download();
  InitSiftData(sift_data, 32768, true, true);
  float *memoryTmp = AllocSiftTempMemory(width_, height_, 5, false);
  ExtractSift(sift_data, img_model_, num_octaves_, init_blur_, thresh_, min_scale_, up_scale_, 
	      memoryTmp);
  FreeSiftTempMemory(memoryTmp);
}

void SiftInterface::uniformFeature(SiftData& sift_data){
  
  mask_ = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(255));
  int data_size = sift_data.numPts;
  
#ifdef MANAGEDMEM
  SiftPoint *sift1 = sift_data.m_data;
#else
  SiftPoint *sift1 = sift_data.h_data;
#endif
  int count = 0;
  for (int i = 0; i < data_size; i++) {
    cv::Point2f pt(sift1[i].xpos, sift1[i].ypos);
    if (mask_.at<uchar>(pt) == 255){
      count++;
      cv::circle(mask_, pt, MIN_DIST_, 0, -1);
      sift1[count] = sift1[i];
      count++;
    }
  }
  sift_data.numPts = count;
  std::cout << "before uniform data size = " << data_size << std::endl;
  std::cout << "after uniform data size = " << count << std::endl;
}

std::vector<cv::KeyPoint> SiftInterface::getKeyPoints(SiftData &sift_data) {
  int data_size = sift_data.numPts;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = sift_data.m_data;
#else
  SiftPoint *sift1 = sift_data.h_data;
#endif
  std::vector<cv::KeyPoint> key_points;
  // std::cout << "data_size = " << data_size << std::endl;

  for (int i = 0; i < data_size; i++) {
    cv::KeyPoint temp_kp;
    temp_kp.pt = cv::Point2f(sift1[i].xpos, sift1[i].ypos);
    temp_kp.angle = sift1[i].orientation;
    temp_kp.response = sift1[i].score;
    temp_kp.octave = sift1[i].scale;
    key_points.push_back(temp_kp);
  }
  return key_points;
}

Matches SiftInterface::rejectWithF(const Matches &matches){
  Matches matches_inlier;
  Eigen::Matrix3d k_matrix;
  cv::cv2eigen(k_mat_, k_matrix);
  std::vector<cv::Point> pts1, pts2;
  for (int i = 0; i < matches.size(); i++) {
    pts1.push_back(matches[i].first);
    pts2.push_back(matches[i].second);
  }

  if (pts2.size() < 8)
    return matches;

  cv::Mat status;
  cv::Mat F_mat = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 1.0, 0.99, status);
  status.convertTo(status, CV_32FC1);
  // cv::findEssentialMat(pts1, pts2, k_mat_, RANSAC, 0.99, 1.0, status);
  
  int j = 0;
  for (int i = 0; i < int(matches.size()); i++){
    if (!status.at<float>(i)) continue;
    // check using p2T*F*p1 = 0
    double error = fundmentalError(F_mat, pts1[i], pts2[i]);
    if(error > 1.0) continue;
    matches_inlier.push_back(matches[i]);
  }
 return matches_inlier; 
}

double SiftInterface::fundmentalError(const cv::Mat f_mat, cv::Point2f pt1, cv::Point2f pt2){
  
  Eigen::Matrix3d f;
  cv::cv2eigen<double>(f_mat, f);
  double u1 = (double)pt1.x, v1 = (double) pt1.y;
  double u2 = (double)pt2.x, v2 = (double) pt2.y;
  double error = u2 *(f(0, 0)*u1 + f(0, 1)*v1 + f(0, 2)) + v2*(f(1, 0)*u1 + 
		 f(1, 1) * v1 + f(1, 2)) + (f(2, 0) * u1 + f(2, 1)*v1 + f(2, 2));
  // std::cout << " error = " << error << std::endl;
  return error;
}

Matches SiftInterface::getCoarseMatches(SiftData &sift_data1, SiftData &sift_data2) {
  Matches matches;
  matches.clear();
  MatchSiftData(sift_data1, sift_data2);
  
  int data1_size = sift_data1.numPts;
  SiftPoint *sift1 = sift_data1.h_data;
  SiftPoint *sift2 = sift_data2.h_data;

  for (int i = 0; i < data1_size; i++) {
    int j = sift1[i].match;
    cv::Point2f pre_pt(sift1[i].xpos, sift1[i].ypos);
    cv::Point2f cur_pt(sift2[j].xpos, sift2[j].ypos);
    std::pair<cv::Point2f, cv::Point2f> tmp = std::make_pair(pre_pt, cur_pt);
    matches.push_back(tmp);
  }
  return matches;
}

Matches SiftInterface::matchDoubleCheck(SiftData &sift_data1, SiftData &sift_data2){
  Matches matches;
  matches.clear();

  MatchSiftData(sift_data1, sift_data2);
  
  SiftData sift_data_copy1 = sift_data1;
  SiftData sift_data_copy2 = sift_data2;
  
  MatchSiftData(sift_data_copy2, sift_data_copy1);
   
  int data1_size = sift_data1.numPts;
  SiftPoint *sift1 = sift_data1.h_data;
  SiftPoint *sift2 = sift_data_copy2.h_data;

  for (int i = 0; i < data1_size; i++) {
    int j = sift1[i].match;
    if(sift2[j].match != i) continue;
    
    double u_diff = abs(sift1[i].xpos - sift1[i].match_xpos);
    double v_diff = abs(sift1[i].ypos - sift1[i].match_ypos);
    if(u_diff > 60 || v_diff > 40) continue;
    cv::Point2f pre_pt(sift1[i].xpos, sift1[i].ypos);
    cv::Point2f cur_pt(sift1[i].match_xpos, sift1[i].match_ypos);
    
        
    std::pair<cv::Point2f, cv::Point2f> tmp = std::make_pair(pre_pt, cur_pt);
    matches.push_back(tmp);
  }
  return matches;
}

Matches SiftInterface::getMatchesByHom(SiftData &data, SiftData &data2, float thresh){
  
  MatchSiftData(data, data2);
  Matches matches;
  float homography[9];
  int numMatches;
  FindHomography(data, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0);
  
  std::vector<int> inlier_indexs;
  inlier_indexs.clear();
#ifdef MANAGEDMEM
  SiftPoint *mpts = data.m_data;
#else
  if (data.h_data == NULL) return matches;
  SiftPoint *mpts = data.h_data;
#endif
  float limit = thresh*thresh;
  int numPts = data.numPts;
  cv::Mat M(8, 8, CV_64FC1);
  cv::Mat A(8, 1, CV_64FC1), X(8, 1, CV_64FC1);
  double Y[8];
  for (int i = 0; i < 8; i++) {
    A.at<double>(i, 0) = homography[i] / homography[8];
  }
  for (int loop = 0; loop < 5; loop++) {
    M = cv::Scalar(0.0);
    X = cv::Scalar(0.0);
    for (int i = 0; i < numPts; i++) {
      SiftPoint &pt = mpts[i];
      if (pt.score < 0.0 || pt.ambiguity > 0.8) continue;
      // den => h7*u1 + h8*v1 + h9
      float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0f;
      float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
      float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
      float err = dx*dx + dy*dy;
      float wei = (err < limit ? 1.0f : 0.0f);
      Y[0] = pt.xpos;
      Y[1] = pt.ypos;
      Y[2] = 1.0;
      Y[3] = Y[4] = Y[5] = 0.0;
      Y[6] = - pt.xpos * pt.match_xpos;
      Y[7] = - pt.ypos * pt.match_xpos;
      for (int c = 0; c < 8; c++) {
	for (int r = 0; r < 8; r++) {
          M.at<double>(r,c) += (Y[c] * Y[r] * wei);
	}
      }
      X += (cv::Mat(8,1,CV_64FC1, Y) * pt.match_xpos * wei);
      Y[0] = Y[1] = Y[2] = 0.0;
      Y[3] = pt.xpos;
      Y[4] = pt.ypos; 
      Y[5] = 1.0;
      Y[6] = - pt.xpos * pt.match_ypos;
      Y[7] = - pt.ypos * pt.match_ypos;
      for (int c=0;c<8;c++) {
        for (int r=0;r<8;r++){ 
          M.at<double>(r,c) += (Y[c] * Y[r] * wei);
	}
      }
      X += (cv::Mat(8, 1, CV_64FC1,Y) * pt.match_ypos * wei);
    }
    // M * A = X
    cv::solve(M, X, A, cv::DECOMP_CHOLESKY);
  }
  int numfit = 0;
  for (int i = 0; i < numPts; i++) {
    SiftPoint &pt = mpts[i];
    float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0;
    float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
    float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
    float err = dx*dx + dy*dy;
    if (err<limit) {
      numfit++;
      inlier_indexs.push_back(i);
    }
    pt.match_error = sqrt(err);
  }
  for (int i = 0; i < 8; i++){
    homography[i] = A.at<double>(i);
  }
  homography[8] = 1.0f;
  
  for (int i = 0; i < inlier_indexs.size(); i++) {
    int j = inlier_indexs[i];
    cv::Point2f pre_pt(mpts[j].xpos, mpts[j].ypos);
    cv::Point2f cur_pt(mpts[j].match_xpos, mpts[j].match_ypos);
    std::pair<cv::Point2f, cv::Point2f> tmp = std::make_pair(pre_pt, cur_pt);
    matches.push_back(tmp);
  }
  // std::cout << "matches size = " << matches.size() << std::endl;
  return matches;
}
