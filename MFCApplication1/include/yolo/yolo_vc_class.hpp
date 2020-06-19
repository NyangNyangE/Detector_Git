#pragma once
#ifdef YOLODLL_EXPORTS
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllexport)
#else
#define YOLODLL_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllimport)
#else
#define YOLODLL_API
#endif
#endif

#include <memory>
#include <vector>
#include <deque>
#include <algorithm>
#include "tree.h"

class Classifier {
public:
	YOLODLL_API Classifier(std::string datacfg, std::string, std::string weightfile, int top);
    YOLODLL_API void hierarchy_predictions_custom(float *predictions, int n, tree *hier, int only_leaves);
    YOLODLL_API std::string run_classification(std::string path_name);
    YOLODLL_API void runrun();
};