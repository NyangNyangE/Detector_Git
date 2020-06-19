#include "yolo_vc_class.hpp"

#include "network.h"

extern "C" {
#include "classifier.h"
#include "parser.h"
#include "image.h"
#include "option_list.h"
#include "utils.h"
#include "tree.h"

#include "blas.h"
#include "assert.h"
#include "cuda.h"
}


network net;
int size;
float* predictions;
list* options;
int top = 0;
YOLODLL_API Classifier::Classifier(std::string datacfg_name, std::string cfgfile_name , std::string weightfile_name, int top) 
{
	char* datacfg = const_cast<char *>(datacfg_name.data());
	char* cfgfile = const_cast<char *>(cfgfile_name.data());
	char* weightfile = const_cast<char *>(weightfile_name.data());

	net = parse_network_cfg_custom(cfgfile, 1);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(2222222);

	size = net.w;
	options = read_data_cfg(datacfg);
	
}

YOLODLL_API void Classifier::hierarchy_predictions_custom(float * predictions, int n, tree * hier, int only_leaves)
{
    int j;
    for (j = 0; j < n; ++j) {
        int parent = hier->parent[j];
        if (parent >= 0) {
            predictions[j] *= predictions[parent];
        }
    }
    if (only_leaves) {
        for (j = 0; j < n; ++j) {
            if (!hier->leaf[j]) predictions[j] = 0;
        }
    }
}

YOLODLL_API std::string Classifier::run_classification(std::string path_name)
{
    char* path = const_cast<char *>(path_name.data());
	char *name_list = option_find_str(options, "names", 0);
	if (!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
	int classes = option_find_int(options, "classes", 2);
	if (top == 0) top = option_find_int(options, "top", 1);
	if (top > classes) top = classes;

	char **names = get_labels(name_list);
	clock_t time;
	int *indexes = (int*)calloc(top, sizeof(int));
	char buff[256];
	char *input = buff;
	size = net.w;
	
	image im = load_image_color(path, 0, 0);
	image r = letterbox_image(im, net.w, net.h);
	
	float *X = r.data;
	predictions = network_predict(net, X);
	if (net.hierarchy) hierarchy_predictions_custom(predictions, net.outputs, net.hierarchy, 0);
	top_k(predictions, net.outputs, top, indexes);

    std::string returnValue = "";
	for (int i = 0; i < top; i++)
	{
		int index = indexes[i];
		if (net.hierarchy) printf("%d, %s: %f, parent: %s \n", index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
		else{
			//printf("%s: %f\n", names[index], predictions[index]);
            returnValue += names[index] ;
		}
		if (r.data != im.data) free_image(r);
		free_image(im);
	}
    return returnValue;
}

YOLODLL_API void Classifier::runrun()
{
    return;
}