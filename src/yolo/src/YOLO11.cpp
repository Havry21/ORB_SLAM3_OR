
#include "det/YOLO11.hpp"

std::vector<std::string> utils::getClassNames(const std::string& path)
{
    std::vector<std::string> classNames;
    std::ifstream infile(path);

    if (infile)
    {
        std::string line;
        while (getline(infile, line))
        {
            // Remove carriage return if present (for Windows compatibility)
            if (! line.empty() && line.back() == '\r')
                line.pop_back();
            classNames.emplace_back(line);
        }
    }
    else
    {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    DEBUG_PRINT("Loaded " << classNames.size() << " class names from " + path);
    return classNames;
}

BoundingBox utils::scaleCoords(const cv::Size& imageShape, BoundingBox coords,
                               const cv::Size& imageOriginalShape, bool p_Clip)
{
    BoundingBox result;
    float gain = std::min(static_cast<float>(imageShape.height) / static_cast<float>(imageOriginalShape.height),
                          static_cast<float>(imageShape.width) / static_cast<float>(imageOriginalShape.width));

    int padX = static_cast<int>(std::round((imageShape.width - imageOriginalShape.width * gain) / 2.0f));
    int padY = static_cast<int>(std::round((imageShape.height - imageOriginalShape.height * gain) / 2.0f));

    result.x = static_cast<int>(std::round((coords.x - padX) / gain));
    result.y = static_cast<int>(std::round((coords.y - padY) / gain));
    result.width = static_cast<int>(std::round(coords.width / gain));
    result.height = static_cast<int>(std::round(coords.height / gain));

    if (p_Clip)
    {
        result.x = utils::clamp(result.x, 0, imageOriginalShape.width);
        result.y = utils::clamp(result.y, 0, imageOriginalShape.height);
        result.width = utils::clamp(result.width, 0, imageOriginalShape.width - result.x);
        result.height = utils::clamp(result.height, 0, imageOriginalShape.height - result.y);
    }
    return result;
}

/**
 * @brief Performs Non-Maximum Suppression (NMS) on the bounding boxes.
 *
 * @param boundingBoxes Vector of bounding boxes.
 * @param scores Vector of confidence scores corresponding to each bounding box.
 * @param scoreThreshold Confidence threshold to filter boxes.
 * @param nmsThreshold IoU threshold for NMS.
 * @param indices Output vector of indices that survive NMS.
 */
// Optimized Non-Maximum Suppression Function
void utils::NMSBoxes(const std::vector<BoundingBox>& boundingBoxes,
                     const std::vector<float>& scores,
                     float scoreThreshold,
                     float nmsThreshold,
                     std::vector<int>& indices)
{
    indices.clear();

    const size_t numBoxes = boundingBoxes.size();
    if (numBoxes == 0)
    {
        DEBUG_PRINT("No bounding boxes to process in NMS");
        return;
    }

    // Step 1: Filter out boxes with scores below the threshold
    // and create a list of indices sorted by descending scores
    std::vector<int> sortedIndices;
    sortedIndices.reserve(numBoxes);
    for (size_t i = 0; i < numBoxes; ++i)
    {
        if (scores[i] >= scoreThreshold)
        {
            sortedIndices.push_back(static_cast<int>(i));
        }
    }

    // If no boxes remain after thresholding
    if (sortedIndices.empty())
    {
        DEBUG_PRINT("No bounding boxes above score threshold");
        return;
    }

    // Sort the indices based on scores in descending order
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&scores](int idx1, int idx2) {
                  return scores[idx1] > scores[idx2];
              });

    // Step 2: Precompute the areas of all boxes
    std::vector<float> areas(numBoxes, 0.0f);
    for (size_t i = 0; i < numBoxes; ++i)
    {
        areas[i] = boundingBoxes[i].width * boundingBoxes[i].height;
    }

    // Step 3: Suppression mask to mark boxes that are suppressed
    std::vector<bool> suppressed(numBoxes, false);

    // Step 4: Iterate through the sorted list and suppress boxes with high IoU
    for (size_t i = 0; i < sortedIndices.size(); ++i)
    {
        int currentIdx = sortedIndices[i];
        if (suppressed[currentIdx])
        {
            continue;
        }

        // Select the current box as a valid detection
        indices.push_back(currentIdx);

        const BoundingBox& currentBox = boundingBoxes[currentIdx];
        const float x1_max = currentBox.x;
        const float y1_max = currentBox.y;
        const float x2_max = currentBox.x + currentBox.width;
        const float y2_max = currentBox.y + currentBox.height;
        const float area_current = areas[currentIdx];

        // Compare IoU of the current box with the rest
        for (size_t j = i + 1; j < sortedIndices.size(); ++j)
        {
            int compareIdx = sortedIndices[j];
            if (suppressed[compareIdx])
            {
                continue;
            }

            const BoundingBox& compareBox = boundingBoxes[compareIdx];
            const float x1 = std::max(x1_max, static_cast<float>(compareBox.x));
            const float y1 = std::max(y1_max, static_cast<float>(compareBox.y));
            const float x2 = std::min(x2_max, static_cast<float>(compareBox.x + compareBox.width));
            const float y2 = std::min(y2_max, static_cast<float>(compareBox.y + compareBox.height));

            const float interWidth = x2 - x1;
            const float interHeight = y2 - y1;

            if (interWidth <= 0 || interHeight <= 0)
            {
                continue;
            }

            const float intersection = interWidth * interHeight;
            const float unionArea = area_current + areas[compareIdx] - intersection;
            const float iou = (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;

            if (iou > nmsThreshold)
            {
                suppressed[compareIdx] = true;
            }
        }
    }

    DEBUG_PRINT("NMS completed with " + std::to_string(indices.size()) + " indices remaining");
}

// Implementation of YOLO11Detector constructor
YOLO11Detector::YOLO11Detector(const std::string& modelPath, const std::string& labelsPath, bool useGPU)
{
    // Initialize ONNX Runtime environment with warning level
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    // Set number of intra-op threads for parallelism
    sessionOptions.SetIntraOpNumThreads(std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Retrieve available execution providers (e.g., CPU, CUDA)
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    // Configure session options based on whether GPU is to be used and available
    if (useGPU && cudaAvailable != availableProviders.end())
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption); // Append CUDA execution provider
    }
    else
    {
        if (useGPU)
        {
            std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        }
        std::cout << "Inference device: CPU" << std::endl;
    }

    session = Ort::Session(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    // Retrieve input tensor shape information
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShapeVec = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    isDynamicInputShape = (inputTensorShapeVec.size() >= 4) && (inputTensorShapeVec[2] == -1 && inputTensorShapeVec[3] == -1); // Check for dynamic dimensions

    // Allocate and store input node names
    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    // Allocate and store output node names
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

    // Set the expected input image shape based on the model's input tensor
    if (inputTensorShapeVec.size() >= 4)
    {
        inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]), static_cast<int>(inputTensorShapeVec[2]));
    }
    else
    {
        throw std::runtime_error("Invalid input tensor shape.");
    }

    // Get the number of input and output nodes
    numInputNodes = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    // Load class names and generate corresponding colors
    classNames = utils::getClassNames(labelsPath);
    classColors = utils::generateColors(classNames);

    std::cout << "Model loaded successfully with " << numInputNodes << " input nodes and " << numOutputNodes << " output nodes." << std::endl;
}

// Preprocess function implementation
cv::Mat YOLO11Detector::preprocess(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    ScopedTimer timer("preprocessing");

    cv::Mat resizedImage;
    // Resize and pad the image using letterBox utility
    utils::letterBox(image, resizedImage, inputImageShape, cv::Scalar(114, 114, 114), isDynamicInputShape, false, true, 32);

    // Update input tensor shape based on resized image dimensions
    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    // Convert image to float and normalize to [0, 1]
    resizedImage.convertTo(resizedImage, CV_32FC3, 1 / 255.0f);

    // Allocate memory for the image blob in CHW format
    blob = new float[resizedImage.cols * resizedImage.rows * resizedImage.channels()];

    // Split the image into separate channels and store in the blob
    std::vector<cv::Mat> chw(resizedImage.channels());
    for (int i = 0; i < resizedImage.channels(); ++i)
    {
        chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1, blob + i * resizedImage.cols * resizedImage.rows);
    }
    cv::split(resizedImage, chw); // Split channels into the blob

    DEBUG_PRINT("Preprocessing completed")

    return resizedImage;
}
// Postprocess function to convert raw model output into detections
std::vector<Detection> YOLO11Detector::postprocess(
    const cv::Size& originalImageSize,
    const cv::Size& resizedImageShape,
    const std::vector<Ort::Value>& outputTensors,
    float confThreshold,
    float iouThreshold)
{
    ScopedTimer timer("postprocessing"); // Measure postprocessing time

    std::vector<Detection> detections;
    const float* rawOutput = outputTensors[0].GetTensorData<float>(); // Extract raw output data from the first output tensor
    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // Determine the number of features and detections
    const size_t num_features = outputShape[1];
    const size_t num_detections = outputShape[2];

    // Early exit if no detections
    if (num_detections == 0)
    {
        return detections;
    }

    // Calculate number of classes based on output shape
    const int numClasses = static_cast<int>(num_features) - 4;
    if (numClasses <= 0)
    {
        // Invalid number of classes
        return detections;
    }

    // Reserve memory for efficient appending
    std::vector<BoundingBox> boxes;
    boxes.reserve(num_detections);
    std::vector<float> confs;
    confs.reserve(num_detections);
    std::vector<int> classIds;
    classIds.reserve(num_detections);
    std::vector<BoundingBox> nms_boxes;
    nms_boxes.reserve(num_detections);

    // Constants for indexing
    const float* ptr = rawOutput;

    for (size_t d = 0; d < num_detections; ++d)
    {
        // Extract bounding box coordinates (center x, center y, width, height)
        float centerX = ptr[0 * num_detections + d];
        float centerY = ptr[1 * num_detections + d];
        float width = ptr[2 * num_detections + d];
        float height = ptr[3 * num_detections + d];

        // Find class with the highest confidence score
        int classId = -1;
        float maxScore = -FLT_MAX;
        for (int c = 0; c < numClasses; ++c)
        {
            const float score = ptr[d + (4 + c) * num_detections];
            if (score > maxScore)
            {
                maxScore = score;
                classId = c;
            }
        }

        // Proceed only if confidence exceeds threshold
        if (maxScore > confThreshold)
        {
            // Convert center coordinates to top-left (x1, y1)
            float left = centerX - width / 2.0f;
            float top = centerY - height / 2.0f;

            // Scale to original image size
            BoundingBox scaledBox = utils::scaleCoords(
                resizedImageShape,
                BoundingBox(left, top, width, height),
                originalImageSize,
                true);

            // Round coordinates for integer pixel positions
            BoundingBox roundedBox;
            roundedBox.x = std::round(scaledBox.x);
            roundedBox.y = std::round(scaledBox.y);
            roundedBox.width = std::round(scaledBox.width);
            roundedBox.height = std::round(scaledBox.height);

            // Adjust NMS box coordinates to prevent overlap between classes
            BoundingBox nmsBox = roundedBox;
            nmsBox.x += classId * 7680; // Arbitrary offset to differentiate classes
            nmsBox.y += classId * 7680;

            // Add to respective containers
            nms_boxes.emplace_back(nmsBox);
            boxes.emplace_back(roundedBox);
            confs.emplace_back(maxScore);
            classIds.emplace_back(classId);
        }
    }

    // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
    std::vector<int> indices;
    utils::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);

    // Collect filtered detections into the result vector
    detections.reserve(indices.size());
    for (const int idx : indices)
    {
        detections.emplace_back(Detection {
            boxes[idx],   // Bounding box
            confs[idx],   // Confidence score
            classIds[idx] // Class ID
        });
    }

    DEBUG_PRINT("Postprocessing completed") // Debug log for completion

    return detections;
}

// Detect function implementation
std::vector<Detection> YOLO11Detector::detect(const cv::Mat& image, int classId, float confThreshold, float iouThreshold)
{
    ScopedTimer timer("Overall detection");

    float* blobPtr = nullptr; // Pointer to hold preprocessed image data
    // Define the shape of the input tensor (batch size, channels, height, width)
    std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height, inputImageShape.width};

    // Preprocess the image and obtain a pointer to the blob
    cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);

    // Compute the total number of elements in the input tensor
    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    // Create a vector from the blob data for ONNX Runtime input
    std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);

    delete[] blobPtr; // Free the allocated memory for the blob

    // Create an Ort memory info object (can be cached if used repeatedly)
    static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor object using the preprocessed data
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorSize,
        inputTensorShape.data(),
        inputTensorShape.size());

    // Run the inference session with the input tensor and retrieve output tensors
    std::vector<Ort::Value> outputTensors = session.Run(
        Ort::RunOptions {nullptr},
        inputNames.data(),
        &inputTensor,
        numInputNodes,
        outputNames.data(),
        numOutputNodes);

    // Determine the resized image shape based on input tensor shape
    cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]), static_cast<int>(inputTensorShape[2]));

    // Postprocess the output tensors to obtain detections
    std::vector<Detection> detections = postprocess(image.size(), resizedImageShape, outputTensors, confThreshold, iouThreshold);

    if (classId != -1)
    {
        std::vector<Detection> detectionForClass;
        for (auto& det : detections)
        {
            if (det.classId == classId)
            {
                detectionForClass.push_back(det);
            }
        }
        return detectionForClass;
    }

    return detections; // Return the vector of detections
}
