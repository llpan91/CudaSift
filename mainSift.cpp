//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//  

#include <cstdio>
#include <cstring>
#include <algorithm>


#include <iostream>  
#include <cmath>
#include <iomanip>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

#include "tic_toc.h"

using std::pair;
using namespace cv;

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, 
		      float maxAmbiguity, float thresh,  std::vector<int>& inlier_indexs);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

void visFeatureTracking(cv::Mat pre_img, cv::Mat cur_img, std::vector<pair<cv::Point2f, cv::Point2f> >& matches);
void getCorrespondence(SiftData &siftData1, SiftData &siftData2, std::vector<int> inlier_indexs,
		       std::vector<std::pair<cv::Point2f, cv::Point2f> > &correspondence);
double ScaleUp(CudaImage &res, CudaImage &src);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  int devNum = 0, imgSet = 0;
  
  TicToc t_start;
  // read two image
  cv::Mat limg, rimg;
  std::string str_img1 = argv[1];
  std::string str_img2 = argv[2];
  cv::imread(str_img1, 0).convertTo(limg, CV_32FC1);
  cv::imread(str_img2, 0).convertTo(rimg, CV_32FC1);
  std::cout << "Image initial cost = " << t_start.toc()  << "ms" << std::endl;
  
  TicToc t_start_11;
  cv::Mat img_pre = cv::imread(str_img1);
  cv::Mat img_cur = cv::imread(str_img2);
  std::cout << "Image cv::imread cost = " << t_start_11.toc()  << "ms" << std::endl;
  
  unsigned int w = limg.cols;
  unsigned int h = limg.rows;

  InitCuda(devNum); 
  TicToc t_start1;
  CudaImage img1(w, h,  iAlignUp(w, 128), false, NULL); 
  CudaImage img2(w, h,  iAlignUp(w, 128), false, NULL); 
  std::cout << "Time1 cost = " << t_start1.toc()  << "ms" << std::endl;
  
  img1.setImageIntiGPU((float*)limg.data);
  img2.setImageIntiGPU((float*)rimg.data);
  img1.Download();
  img2.Download(); 
  
  TicToc t_start2;
  // Extract Sift features from images
  SiftData siftData1, siftData2;
  float initBlur = 1.0f;
  float thresh = (imgSet ? 4.5f : 3.0f);
  InitSiftData(siftData1, 32768, true, true); 
  InitSiftData(siftData2, 32768, true, true);
  float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
  std::cout << "Time2 cost = " << t_start2.toc()  << "ms" << std::endl;
  
  TicToc t_start3;
  ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f, false, memoryTmp);
  ExtractSift(siftData2, img2, 5, initBlur, thresh, 0.0f, false, memoryTmp);
  FreeSiftTempMemory(memoryTmp);
  std::cout << "Time3 cost = " << t_start3.toc()  << "ms" << std::endl;
  // Match Sift features and find a homography
  
  TicToc t_start4;
  MatchSiftData(siftData1, siftData2);
  
  float homography[9];
  int numMatches;
  std::vector<int> inlier_indexs;
  FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0);
  int numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 3.0, inlier_indexs);
  std::cout << "Time4 cost = " << t_start4.toc()  << "ms" << std::endl;
  
  std::cout << "Number of original features: " <<  siftData1.numPts << " " << siftData2.numPts << std::endl;
  std::cout << "Number of matching features: " << numFit << " " << std::endl;
  std::cout << "matching rate = " << 100.0f*numFit/std::min(siftData1.numPts, siftData2.numPts) << "% " << std::endl;
  
  // PrintMatchData(siftData1, siftData2, img1);
  std::vector<pair<cv::Point2f, cv::Point2f> > correspondence;
  getCorrespondence(siftData1, siftData2, inlier_indexs, correspondence);
  visFeatureTracking(img_pre, img_cur, correspondence);
  
  MatchAll(siftData1, siftData2, homography);
  
  // Free Sift data from device
  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
  
//   std::cout << "time = " << t_start.toc() << std::endl;
}

void visFeatureTracking(cv::Mat pre_img, cv::Mat cur_img, std::vector<pair<cv::Point2f, cv::Point2f> >& matches){
  
  std::cout << "start viz feature " << std::endl;
  cv::Scalar tracked(0, 255, 0);
  cv::Scalar new_feature(0, 255, 255);

  int img_height = pre_img.rows;
  int img_width = pre_img.cols;
  cv::Mat out_img(img_height, img_width * 2, CV_8UC3);
  cv::hconcat(pre_img, cur_img, out_img);

  // Draw tracked features.
  const int inlier_size = matches.size();
  for (int idx = 0; idx < inlier_size; idx++) {
    cv::Point2f prev_pt = matches[idx].first;
    cv::Point2f curr_pt = matches[idx].second + cv::Point2f(img_width, 0.0);
    cv::circle(out_img, prev_pt, 3, tracked, -1);
    cv::circle(out_img, curr_pt, 3, tracked, -1);
    cv::line(out_img, prev_pt, curr_pt, tracked, 1);
  }

  // int index = ptr_pre_frame.getId();
  static int index = 0;
  index++;
  // std::string str_index = std::to_string(index);
  std::string img_name = "match.png";
  char text[100];
  // Display text on images 
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 2;
  int thickness = 2.0;
  cv::Point textOrg(10, 50);
  cv::putText(out_img, text, textOrg, fontFace, fontScale, CV_RGB(255, 0, 0), thickness, 6);
  cv::imwrite(img_name, out_img);
//   cv::imshow("Feature", out_img);
//   cv::waitKey(1000);

  return;
  
}

void getCorrespondence(SiftData &siftData1, SiftData &siftData2, std::vector<int> inlier_indexs,
		       std::vector<pair<cv::Point2f, cv::Point2f> > &correspondence){
  
  correspondence.clear();
  int numPts = siftData1.numPts;
  #ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  // find max 
//   for()
  
  for (int i = 0; i < inlier_indexs.size(); i++) {
    int j = inlier_indexs[i];
    int k = sift1[j].match;
    cv::Point2f pre_pt(sift1[j].xpos, sift1[j].ypos);
    cv::Point2f cur_pt(sift2[k].xpos, sift2[k].ypos);
    
    std::pair<cv::Point2f, cv::Point2f> tmp = std::make_pair(pre_pt, cur_pt);
    correspondence.push_back(tmp);
  }
  std::cout << "matches size = " << correspondence.size() << std::endl;
}

// 垃圾代码 小学生水平
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography){
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;
  int numFound = 0;

  for (int i = 0; i < numPts1; i++) {
    float *data1 = sift1[i].data;
    std::cout << i << ":" << sift1[i].scale << ":" << (int)sift1[i].orientation << " " << sift1[i].xpos << " " << sift1[i].ypos << std::endl;
    bool found = false;
    for (int j = 0;j < numPts2; j++) {
      float *data2 = sift2[j].data;
      float sum = 0.0f;
      for (int k = 0; k < 128; k++) sum += data1[k]*data2[k];    
      float den = homography[6]*sift1[i].xpos + homography[7]*sift1[i].ypos + homography[8];
      float dx = (homography[0]*sift1[i].xpos + homography[1]*sift1[i].ypos + homography[2]) / den - sift2[j].xpos;
      float dy = (homography[3]*sift1[i].xpos + homography[4]*sift1[i].ypos + homography[5]) / den - sift2[j].ypos;
      float err = dx*dx + dy*dy;
      if (err < 100.0f) // 100.0
	found = true;
      if (err < 100.0f || j == sift1[i].match) { // 100.0
	if (j == sift1[i].match && err < 100.0f) std::cout << " *";
	else if (j == sift1[i].match) std::cout << " -";
	else if (err<100.0f) std::cout << " +";
	else std::cout << "  ";
	std::cout << j << ":" << sum << ":" << (int)sqrt(err) << ":" << sift2[j].scale << ":" << (int)sift2[j].orientation << " " << sift2[j].xpos << " " << sift2[j].ypos << " " << (int)dx << " " << (int)dy << std::endl;
      }
    }
    std::cout << std::endl;
    if (found)
      numFound++;
  }
  std::cout << "Number of finds: " << numFound << " / " << numPts1 << std::endl;
  std::cout << homography[0] << " " << homography[1] << " " << homography[2] << std::endl;//%%%
  std::cout << homography[3] << " " << homography[4] << " " << homography[5] << std::endl;//%%%
  std::cout << homography[6] << " " << homography[7] << " " << homography[8] << std::endl;//%%%
}

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img){
  int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  float *h_img = img.h_data;
  int w = img.width;
  int h = img.height;
  std::cout << std::setprecision(3);
  for (int j = 0; j < numPts; j++) {
    int k = sift1[j].match;
    if (sift1[j].match_error < 5) {
      float dx = sift2[k].xpos - sift1[j].xpos;
      float dy = sift2[k].ypos - sift1[j].ypos;
      int len = (int)(fabs(dx) > fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l = 0; l < len; l++) {
	int x = (int)(sift1[j].xpos + dx*l/len);
	int y = (int)(sift1[j].ypos + dy*l/len);
	h_img[y*w+x] = 255.0f;
      }
    }
    // std::cout << "j-th coordinate " << sift1[j].xpos << ", " << sift1[j].ypos << std::endl;
    // std::cout << "k-th coordinate " << sift2[k].xpos << ", " << sift2[k].ypos << std::endl;
    
    int x = (int)(sift1[j].xpos+0.5);
    int y = (int)(sift1[j].ypos+0.5);
    int s = std::min(x, std::min(y, std::min(w-x-2, std::min(h-y-2, (int)(1.41*sift1[j].scale)))));
    int p = y*w + x;
    p += (w+1);
    for (int k = 0; k < s; k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] = h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 255.0f;
  }
  std::cout << std::setprecision(6);
}


