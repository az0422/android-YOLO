package org.tensorflow.lite.examples.detection.tflite;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

import org.tensorflow.lite.examples.detection.env.Utils;

public class YoloClassifier implements Classifier {
    private int INPUT_SIZE;
    private float YOLO_VERSION;
    private int OUTPUT_SHAPE;
    private Vector<String> labels = new Vector<>();

    private Interpreter TFLITE;

    public YoloClassifier(final AssetManager assetManager,
                          final String modelFilename,
                          final String labelFilename,
                          final int input_size,
                          final int output_shape,
                          final float yolo_version) throws IOException {

        INPUT_SIZE = input_size;
        YOLO_VERSION = yolo_version;
        OUTPUT_SHAPE = output_shape;

        InputStream labelInput = assetManager.open(labelFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelInput));

        String line;

        while ((line = br.readLine()) != null) {
            Log.i("Read label", line);
            labels.add(line);
        }
        br.close();

        try {
            Interpreter.Options options = new Interpreter.Options();
            CompatibilityList compatList = new CompatibilityList();
            options.setUseXNNPACK(true);
            options.setUseNNAPI(true);
            options.setNumThreads(4);

            GpuDelegateFactory.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);

//            if(compatList.isDelegateSupportedOnThisDevice()){
//                // if the device has a supported GPU, add the GPU delegate
//                GpuDelegateFactory.Options delegateOptions = compatList.getBestOptionsForThisDevice();
//                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
//                options.addDelegate(gpuDelegate);
//            } else {
//                // if the GPU is not supported, run on 4 threads
//                options.setNumThreads(NUM_THREADS);
//                options.setUseNNAPI(true);
//            }

            TFLITE = new Interpreter(Utils.loadModelFile(assetManager, modelFilename), options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    protected float mNmsThresh = 0.75f;

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<Recognition>();

        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            50,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected static final int BATCH_SIZE = 1;
    protected static final int PIXEL_SIZE = 3;

    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }

    private ArrayList<Recognition> recognizeImageV4(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][OUTPUT_SHAPE][4]);
        outputMap.put(1, new float[1][OUTPUT_SHAPE][labels.size()]);
        Object[] inputArray = {byteBuffer};
        TFLITE.runForMultipleInputsOutputs(inputArray, outputMap);

        int gridWidth = OUTPUT_SHAPE;
        float[][][] bboxes = (float [][][]) outputMap.get(0);
        float[][][] out_score = (float[][][]) outputMap.get(1);

        for (int i = 0; i < gridWidth; i++){
            float maxClass = 0;
            int detectedClass = -1;
            final float[] classes = new float[labels.size()];

            for (int c = 0;c< labels.size();c++){
                classes [c] = out_score[0][i][c];
            }

            for (int c = 0;c<labels.size();++c){
                if (classes[c] > maxClass){
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }
            final float score = maxClass;
            if (score > getObjThresh()){
                final float xPos = bboxes[0][i][0];
                final float yPos = bboxes[0][i][1];
                final float w = bboxes[0][i][2];
                final float h = bboxes[0][i][3];
                final RectF rectF = new RectF(
                        Math.max(0, xPos - w / 2),
                        Math.max(0, yPos - h / 2),
                        Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                        Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + i, labels.get(detectedClass),score,rectF,detectedClass ));
            }
        }
        return nms(detections);
    }

    private List<Recognition> recognizeImageV5(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();

        //outputMap.put(0, ByteBuffer.allocateDirect(1 * OUTPUT_SHAPE[0] * (labels.size() + 5) * 4));
        outputMap.put(0, new float[1][OUTPUT_SHAPE][5 + labels.size()]);
        Object[] inputArray = {byteBuffer};

        Log.i("null reference", "" + outputMap + " " + inputArray);

        TFLITE.runForMultipleInputsOutputs(inputArray, outputMap);

        float[][][] out = (float[][][])outputMap.get(0);

//        Log.i("outputshape", "" + out[0].length);

//        for (int i = 0; i < OUTPUT_SHAPE[0]; i++) {
//            Log.i("outputshape-i", "" + i);
//            for (int j = 0; j < 4; j++) {
//                out[0][i][j] *= INPUT_SIZE;
//            }
//        }

        for (int i = 0; i < OUTPUT_SHAPE; i++) {
            final int offset = 0;
            final float confidence = out[0][i][4];
            int detectedClass = -1;
            float maxClass = 0;

            final float[] classes = new float[labels.size()];

            for (int c = 0; c < labels.size(); c++) {
                classes[c] = out[0][i][5 + c];
            }

            for (int c = 0; c < labels.size(); c++) {
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }

            final float confidenceInClass = maxClass * confidence;
            if (confidenceInClass > getObjThresh()) {
                final float xPos = out[0][i][0] * INPUT_SIZE;
                final float yPos = out[0][i][1] * INPUT_SIZE;

                final float w = out[0][i][2] * INPUT_SIZE;
                final float h = out[0][i][3] * INPUT_SIZE;
                Log.d("YoloV5Classifier",
                        Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h);

                final RectF rect =
                        new RectF(
                                Math.max(0, xPos - w / 2),
                                Math.max(0, yPos - h / 2),
                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + offset, labels.get(detectedClass),
                        confidenceInClass, rect, detectedClass));
            }
        }

        return nms(detections);
    }

    private List<Recognition> recognizeImageV8(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();

        //outputMap.put(0, ByteBuffer.allocateDirect(1 * OUTPUT_SHAPE[0] * (labels.size() + 5) * 4));
        outputMap.put(0, new float[1][4 + labels.size()][OUTPUT_SHAPE]);
        Object[] inputArray = {byteBuffer};

        Log.i("null reference", "" + outputMap + " " + inputArray);

        TFLITE.runForMultipleInputsOutputs(inputArray, outputMap);

        float[][][] out = (float[][][])outputMap.get(0);

        for (int i = 0; i < OUTPUT_SHAPE; i++) {
            final int offset = 0;
            int detectedClass = -1;
            float maxClass = 0;

            final float[] classes = new float[labels.size()];

            for (int c = 0; c < labels.size(); c++) {
                classes[c] = out[0][4 + c][i];
            }

            for (int c = 0; c < labels.size(); c++) {
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }

            final float confidenceInClass = maxClass;
            if (confidenceInClass > getObjThresh()) {
                final float xPos = out[0][0][i];
                final float yPos = out[0][1][i];

                final float w = out[0][2][i];
                final float h = out[0][3][i];

                Log.d("YoloV8Classifier",
                        Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h);

                final RectF rect =
                        new RectF(
                                Math.max(0, xPos - w / 2),
                                Math.max(0, yPos - h / 2),
                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + offset, labels.get(detectedClass),
                        confidenceInClass, rect, detectedClass));
            }
        }

        return nms(detections);
    }

    private List<Recognition> recognizeImageV8r1(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();

        //outputMap.put(0, ByteBuffer.allocateDirect(1 * OUTPUT_SHAPE[0] * (labels.size() + 5) * 4));
        outputMap.put(0, new float[1][4 + labels.size()][OUTPUT_SHAPE]);
        Object[] inputArray = {byteBuffer};

        Log.i("null reference", "" + outputMap + " " + inputArray);

        TFLITE.runForMultipleInputsOutputs(inputArray, outputMap);

        float[][][] out = (float[][][])outputMap.get(0);

        for (int i = 0; i < OUTPUT_SHAPE; i++) {
            final int offset = 0;
            int detectedClass = -1;
            float maxClass = 0;

            final float[] classes = new float[labels.size()];

            for (int c = 0; c < labels.size(); c++) {
                classes[c] = out[0][4 + c][i];
            }

            for (int c = 0; c < labels.size(); c++) {
                if (classes[c] > maxClass && classes[c] != 1.0) {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }

            final float confidenceInClass = maxClass;
            if (confidenceInClass > getObjThresh()) {
                final float xPos = out[0][0][i] * INPUT_SIZE;
                final float yPos = out[0][1][i] * INPUT_SIZE;

                final float w = out[0][2][i] * INPUT_SIZE;
                final float h = out[0][3][i] * INPUT_SIZE;

                Log.d("YoloV8Classifier",
                        Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h + ", (" + confidenceInClass + ")");

                final RectF rect =
                        new RectF(
                                Math.max(0, xPos - w / 2),
                                Math.max(0, yPos - h / 2),
                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + offset, labels.get(detectedClass),
                        confidenceInClass, rect, detectedClass));
            }
        }

        return nms(detections);
    }

    private List<Recognition> recognizeImageV3(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();

        //outputMap.put(0, ByteBuffer.allocateDirect(1 * OUTPUT_SHAPE[0] * (labels.size() + 5) * 4));
        outputMap.put(0, new float[1][OUTPUT_SHAPE / 2][5 + labels.size()]);
        Object[] inputArray = {byteBuffer};

        TFLITE.runForMultipleInputsOutputs(inputArray, outputMap);

        float[][][] out = (float[][][])outputMap.get(0);

//        Log.i("outputshape", "" + out[0].length);

//        for (int i = 0; i < OUTPUT_SHAPE[0]; i++) {
//            Log.i("outputshape-i", "" + i);
//            for (int j = 0; j < 4; j++) {
//                out[0][i][j] *= INPUT_SIZE;
//            }
//        }

        for (int i = 0; i < OUTPUT_SHAPE; i++) {
            final int offset = 0;
            final float confidence = out[0][i][4];
            int detectedClass = -1;
            float maxClass = 0;

            final float[] classes = new float[labels.size()];

            for (int c = 0; c < labels.size(); c++) {
                classes[c] = out[0][i][5 + c];
            }

            for (int c = 0; c < labels.size(); c++) {
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }

            final float confidenceInClass = maxClass * confidence;
            if (confidenceInClass > getObjThresh()) {
                final float xPos = out[0][i][0] * INPUT_SIZE;
                final float yPos = out[0][i][1] * INPUT_SIZE;

                final float w = out[0][i][2] * INPUT_SIZE;
                final float h = out[0][i][3] * INPUT_SIZE;
//                Log.d("YoloV5Classifier",
//                        Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h);

                final RectF rect =
                        new RectF(
                                Math.max(0, xPos - w / 2),
                                Math.max(0, yPos - h / 2),
                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + offset, labels.get(detectedClass),
                        confidenceInClass, rect, detectedClass));
            }
        }

        return nms(detections);
    }

    Context context;

    public void setContext(Context context) {
        this.context = context;
    }

    public List<Recognition> recognizeImage(Bitmap bitmap) {

        if (YOLO_VERSION == 4) {
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
            return recognizeImageV4(byteBuffer, bitmap);
        }
        else if (YOLO_VERSION == 5) {
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
            return recognizeImageV5(byteBuffer, bitmap);
        }
        else if (YOLO_VERSION == 8) {
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
            return recognizeImageV8(byteBuffer, bitmap);
        }
        else if (YOLO_VERSION == 8.1f) {
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
            return recognizeImageV8r1(byteBuffer, bitmap);
        }

        return null;
    }

    @Override
    public void enableStatLogging(boolean debug) {

    }

    @Override
    public String getStatString() {
        return null;
    }

    @Override
    public void close() {

    }

    @Override
    public void setNumThreads(int num_threads) {

    }

    @Override
    public void setUseNNAPI(boolean isChecked) {

    }

    @Override
    public float getObjThresh() {
        return 0.3f;
    }
}
