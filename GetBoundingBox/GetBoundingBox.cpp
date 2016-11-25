// GetBoundingBox.cpp : �w�q�D���x���ε{�����i�J�I�C
//

#include "stdafx.h"
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <string.h>
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <string>
using namespace cv;
using namespace std;
//------------parameter-------------
#define Threshold_param 220
#define Smooth_mask 3  //3x3
#define Canny_threshold_param_down 40
#define Canny_threshold_param_up 200
#define Canny_mask 3
#define BoundingBox_threshold 17
#define BoundingBox_Selected_thes_down 40
#define BoundingBox_Selected_thes_up 70

//#define OPEN_DEBUG 1
//CvMemStorage* storage = NULL;


void BoundingBox(IplImage* img_in, char *outputpath, int BBox_Erode_iter, int BBox_Dilate_iter, int image_100_Dilate, int image_100_Erode)
{
	IplImage* img_working = cvCreateImage(cvGetSize(img_in), 8, 1);
	IplImage* img_a_little_bigger;
	IplImage* draw;
	cvCvtColor(img_in, img_working, CV_BGR2GRAY);
	//--------------------���X�@�ӭ�Ϩ⭿�j�� ���� img_a_little_bigger---------------------
	//--------------------�A���϶K�� img_a_little_bigger ������---------------------------
	/*
	 __________
	|  ______  |
	| |      | |
	| |  2   | |
	| |______| |
	|__________|
	*/
	CvSize little_bigger_cvsize;
	little_bigger_cvsize.width = img_working->width * 2;
	little_bigger_cvsize.height = img_working->height * 2;
	img_a_little_bigger = cvCreateImage(little_bigger_cvsize, img_working->depth, img_working->nChannels);
	cvSet(img_a_little_bigger, cvScalar(255, 255, 255));
	CvRect little_bigger_rect;
	little_bigger_rect.x = (little_bigger_cvsize.width - img_working->width) / 2;
	little_bigger_rect.y = (little_bigger_cvsize.height - img_working->height) / 2;
	little_bigger_rect.width = img_working->width;
	little_bigger_rect.height = img_working->height;
	cvSetImageROI(img_a_little_bigger, little_bigger_rect);
	cvCopy(img_working, img_a_little_bigger);
	cvResetImageROI(img_a_little_bigger);
	draw = cvCloneImage(img_a_little_bigger);
	//-----------------------------------------------------------------------
	//-------------Threshold and inverse image (white->black)----------------
	//-------------�Ȧs�@�� ThresholdImg �H�ѳ̫� crop image��---------------
	CvSeq* seq;
	//CvMemStorage* storage = NULL;
	vector<CvRect> boxes;
	CvMemStorage* storage = cvCreateMemStorage(0);
	cvClearMemStorage(storage);
	IplImage* ThresholdImg = cvCloneImage(img_a_little_bigger);
	cvThreshold(ThresholdImg, ThresholdImg, Threshold_param, 255, THRESH_OTSU);
	cvThreshold(ThresholdImg, ThresholdImg, Threshold_param, 255, THRESH_BINARY_INV);
	//-----------------------------------------------------------------------
	/*
	���F���bounding box �Nimg_a_little_bigger �� 
	1. smooth(median) �h�� salt and pepper
	2. Erode (��խI���¦r���O���D) + Dilate (��խI���¦r���O�ܽG) �ɰ_�զ�Ŭ}
	3. OTSU �G�Ȥ�
	4. Canny �����
	5. FindContours ro get Bounding box
	*/
#ifdef OPEN_DEBUG
	cvShowImage("ThresholdImg THRESH_OTSU", ThresholdImg);
	cvWaitKey(0);
#endif
	cvSmooth(img_a_little_bigger, img_a_little_bigger, CV_MEDIAN, Smooth_mask, 0);
#ifdef OPEN_DEBUG
	cvShowImage("img_working", img_a_little_bigger);
	cvWaitKey(0);
#endif
	cvErode(img_a_little_bigger, img_a_little_bigger, 0, BBox_Erode_iter);  // �nŪ��   20
	cvDilate(img_a_little_bigger, img_a_little_bigger, 0, BBox_Dilate_iter); //10
#ifdef OPEN_DEBUG
	cvShowImage("img_working cvErode cvDilate", img_a_little_bigger);
	cvWaitKey(0);
#endif
	cvThreshold(img_a_little_bigger, img_a_little_bigger, Threshold_param, 255, THRESH_OTSU);
#ifdef OPEN_DEBUG
	cvShowImage("img_working THRESH_OTSU", img_a_little_bigger);
	cvWaitKey(0);
#endif
	cvCanny(img_a_little_bigger, img_a_little_bigger, Canny_threshold_param_down, Canny_threshold_param_up, Canny_mask);
	cvFindContours(img_a_little_bigger, storage, &seq, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	CvRect boundbox;
	int push_count = 0;
	for (; seq; seq = seq->h_next) {
		boundbox = cvBoundingRect(seq);
		if (boundbox.height > BoundingBox_threshold && boundbox.width > BoundingBox_threshold) //BoundingBox �j�p ���e�n�j��17 (���X�Ӫ�)
		{
			boxes.push_back(boundbox); push_count++;
		}
		else if (boxes.size() == 0 && !seq->h_next)
		{
			boxes.push_back(boundbox); push_count++;
		}
	}
	//-------------------------------------------�����preprocessing�n���o�n---------------------------------------------------------------
	//�p�G���2�� bounding box �P�_ bounding box �O�_�b�S�w�d�� (���O�a�����) �p�G�����preprocessing���o�n �h���ݭn���B�J
	//�ثecase���S����W�L2��bounding box
	int choose = 0;
	if (push_count > 1 
		&& boxes[1].x > BoundingBox_Selected_thes_up 
		&& boxes[1].x < img_a_little_bigger->width - BoundingBox_Selected_thes_down
		&& boxes[1].y > BoundingBox_Selected_thes_up
		&& boxes[1].y < img_a_little_bigger->height - BoundingBox_Selected_thes_down)
	{
		choose = 1;
	}
	//-------------------------------------------------------------------------------------------------------------------------------
	//�e�X bounding box
#ifdef OPEN_DEBUG
	cvRectangle(draw, cvPoint(boxes[choose].x, boxes[choose].y),
		cvPoint(boxes[choose].x + boxes[choose].width,
		boxes[choose].y + boxes[choose].height), CV_RGB(255, 0, 0),
		1, 8, 0);
	cvShowImage("Rectangle", draw);
	cvWaitKey(0);
#endif
	//-------------------------------------------------------------------------------------------------------------------------------
	//------------------------------�q�@�}�l�ƻs�@���� ThresholdImg ���BoundingBox ROI and crop it----------------------------------
	CvRect rect;
	IplImage* imgRoi = cvCloneImage(ThresholdImg);
	rect.width = boxes[choose].width;
	rect.height = boxes[choose].height;
	rect.x = boxes[choose].x;
	rect.y = boxes[choose].y;
	//�ŤU boundingbox �ϰ�
	IplImage* cropped = cvCreateImage(cvSize(rect.width, rect.height), imgRoi->depth, imgRoi->nChannels);
	cvSetImageROI(imgRoi, rect);
	cvCopy(imgRoi, cropped);
	cvResetImageROI(imgRoi);
	printf("Get bounding box region\r\n");
#ifdef OPEN_DEBUG
	cvShowImage("cropped", cropped);
	cvWaitKey(0);
#endif

	//�Q��boundingbox��X�Ӫ�image����ҶK�� ���¥����image ��������  (�ܦ�cropped_img_to_square)
	//���¥����image �j�p �H boundingbox��X��image ���̪��� ��@���諬����
	IplImage* cropped_img_to_square;
	CvSize cropped_img_to_square_cvsize;
	int square_size = max(rect.height, rect.width);
	cropped_img_to_square_cvsize.width = square_size * 1;
	cropped_img_to_square_cvsize.height = square_size * 1;
	cropped_img_to_square = cvCreateImage(cropped_img_to_square_cvsize, cropped->depth, cropped->nChannels);
	cvSet(cropped_img_to_square, cvScalar(0, 0, 0));
	CvRect cropped_img_to_square_rect;
	cropped_img_to_square_rect.x = (cropped_img_to_square_cvsize.width - cropped->width) / 2;
	cropped_img_to_square_rect.y = (cropped_img_to_square_cvsize.height - cropped->height) / 2;
	cropped_img_to_square_rect.width = cropped->width;
	cropped_img_to_square_rect.height = cropped->height;
	cvSetImageROI(cropped_img_to_square, cropped_img_to_square_rect);
	cvCopy(cropped, cropped_img_to_square);
	cvResetImageROI(cropped_img_to_square);
#ifdef OPEN_DEBUG
	cvShowImage("cropped_img_to_square_rect", cropped_img_to_square_rect);
	cvWaitKey(0);
#endif
	//------------------------------------��jcropped_img_to_square �� 100x100-----------------------------------------
	IplImage *dst_100x100 = 0, *dst;
	CvSize dst_cvsize;
	dst_cvsize.width = 100;
	dst_cvsize.height = 100;
	dst_100x100 = cvCreateImage(dst_cvsize, cropped_img_to_square->depth, cropped_img_to_square->nChannels);
	cvResize(cropped_img_to_square, dst_100x100, CV_INTER_CUBIC);
#ifdef OPEN_DEBUG
	cvShowImage("dst_100x100", dst_100x100);
	cvWaitKey(0);
#endif
	//-------------Dilate (��­I���զr���O���D) + Erode (��­I���զr���O�ܽG) �ɰ_�@�Ǧ]threshold�ӾɭP���s�򪺦a��---------------
	cvDilate(dst_100x100, dst_100x100, 0, image_100_Dilate); //3
	cvErode(dst_100x100, dst_100x100, 0, image_100_Erode); //3
	printf("finish!\r\n");
	printf(outputpath);
	printf("\r\n");
#ifdef OPEN_DEBUG
	cvShowImage("dst_100x100", dst_100x100);
	cvWaitKey(0);
#endif
	//---------------------------------------------------�Y�p��28x28----------------------------------------------------------------
	dst_cvsize.width = 28;
	dst_cvsize.height = 28;
	dst = cvCreateImage(dst_cvsize, cropped_img_to_square->depth, cropped_img_to_square->nChannels);
	cvResize(dst_100x100, dst, CV_INTER_CUBIC);
#ifdef OPEN_DEBUG
	cvShowImage("dst", dst);
	cvWaitKey(0);
#endif
	cvSaveImage(outputpath, dst);
}


int _tmain(int argc, _TCHAR* argv[])
{
	char readimgname[64] = { 0 };
	char outputimgname[128] = { 0 };
	string file_contents;
	ifstream file("BoundingBoxParam.txt");
	string str;
	int filenumber = 100, BountdingBox_Erode_iter = 0, BountdingBox_Dilate_iter = 0;
	int index = 0, image_100x100_Dilate = 0, image_100x100_Erode = 0;
	int num_index = 0;
	int readfile_count = 0;
	while (std::getline(file, str))
	{
		file_contents = str;
		switch (readfile_count)
		{
			case 0:
				num_index = stoi(file_contents);
				break;
			case 1:
				filenumber = stoi(file_contents);
				break;
			case 2:
				BountdingBox_Erode_iter = stoi(file_contents);
				break;
			case 3:
				BountdingBox_Dilate_iter = stoi(file_contents);
				break;
			case 4:
				image_100x100_Dilate = stoi(file_contents);
				break;
			case 5:
				image_100x100_Erode = stoi(file_contents);
				break;
		}
		readfile_count++;
	}

	for (index = num_index; index <= num_index; index++)
	{
		for (int j = 1; j <= filenumber; j++)
		{
			sprintf(readimgname, "D:\\Skywawtch\\Working\\caffe\\outputROI\\%d\\input_%d (%d).jpg", index, index, j);
			sprintf(outputimgname, "D:\\Skywawtch\\Working\\caffe\\outputROI\\%d\\%d_BoundingBox\\test (%d).jpg", index, index, j);
			IplImage* img_in = cvLoadImage(readimgname, 1);
			BoundingBox(img_in, outputimgname, BountdingBox_Erode_iter, 
				BountdingBox_Dilate_iter, image_100x100_Dilate, image_100x100_Erode);
		}
	}
	return 0;
}

