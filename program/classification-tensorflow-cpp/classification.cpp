/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

// TODO: this header should be moved to a common location (where?)
#include "../classification-tflite-cpp/benchmark.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/framework/scope.h"

using namespace std;
using namespace CK;
using namespace tensorflow;

int main(int argc, char* argv[]) {
  try {
    init_benchmark();
    
    BenchmarkSettings settings;
    BenchmarkSession session(&settings);
    ImageData input_data(&settings);
    ResultData result_data(&settings);
    InNormalize input_converter(&settings);
    OutCopy result_converter(&settings);
    unique_ptr<Session> tf_session;
    GraphDef graph_def;

    const string input_layer_name = getenv_s("RUN_OPT_INPUT_LAYER_NAME");
    const string output_layer_name = getenv_s("RUN_OPT_OUTPUT_LAYER_NAME");
    // TODO: this option is for TF mobilenets, but generally should be evaluated
    // from weights package somehow (supported number or classes in meta?)
    // TODO: this problem is related to the absence of a knowledge about
    // required image size for particular image recognition network package.
    // TODO: We have to provide common set of parameters for all image-recognition packages.
    const bool has_background_class = true; 

    cout << "\nLoading graph..." << endl;
    measure_setup([&]{
      Status status = ReadBinaryProto(Env::Default(), settings.graph_file, &graph_def);
      if (!status.ok())
        throw "Failed to load graph: " + status.ToString();

      tf_session.reset(NewSession(SessionOptions()));

      status = tf_session->Create(graph_def);
      if (!status.ok())
        throw "Failed to create new session: " + status.ToString();
    });

    cout << "\nProcessing batches..." << endl;
    measure_prediction([&]{
      Tensor input(DT_FLOAT, TensorShape({settings.batch_size,
                                          settings.image_size,
                                          settings.image_size,
                                          settings.num_channels}));
      float* input_ptr = input.flat<float>().data();
      vector<Tensor> outputs;

      while (session.get_next_batch()) {
        // Load batch
        session.measure_begin();
        int image_offset = 0;
        for (auto image_file : session.batch_files()) {
          input_data.load(image_file);
          input_converter.convert(&input_data, input_ptr + image_offset);
          image_offset += input_data.size();
        }
        session.measure_end_load_images();

        // Classify current batch
        session.measure_begin();
        Status status = tf_session->Run(
          {{input_layer_name, input}}, {output_layer_name}, {}, &outputs);
        if (!status.ok())
          throw "Running model failed: " + status.ToString();
        session.measure_end_prediction();

        // Process output tensor
        auto output_flat = outputs[0].flat<float>();
        if (output_flat.size() != settings.batch_size * (settings.num_classes + 1))
          throw format("Output tensor has size of %d, but expected size is %d",
                       output_flat.size(), settings.batch_size * (settings.num_classes + 1));
        image_offset = 0;
        int probe_offset = has_background_class ? 1 : 0;
        for (auto image_file : session.batch_files()) {
          result_converter.convert(output_flat.data() + image_offset + probe_offset, &result_data);
          result_data.save(image_file);
          image_offset += result_data.size() + probe_offset;
        }
      }
    });

    finish_benchmark(session);
  }
  catch (const string& error_message) {
    cerr << "ERROR: " << error_message << endl;
    return -1;
  }
  return 0;
}
