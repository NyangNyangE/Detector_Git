
list *read_data_cfg(char *filename);
void run_classifier(int argc, char **argv);
void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
