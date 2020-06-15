const axios = require('axios');
const tf = require('@tensorflow/tfjs-node');

// Essa função vai servir para baixarmos a imagem que queremos analisar
const getImageBuffer = async (url) => {
    const response = await axios.get(url, { responseType: 'arraybuffer' });
    return Buffer.from(response.data);
};

// Essas são as labels na ordem que o CustomVision gerou
const PREDICT_LABELS = {
    0: 'Abnormal',
    1: 'Normal'
};

// Um reduce simples para nomear os resultados mostrando as respectivas labels
const applyLabels = (predictions) => (
    predictions.reduce((prev, next, index) => ({ ...prev, [PREDICT_LABELS[index]]: next }), {})
);

const predict = async () => {
    // Carregando o model gerado no TensorFlow
    const model = await tf.loadGraphModel(tf.io.fileSystem(__dirname + '/model/model.json'));

    // Baixando a imagem
    const buf = await getImageBuffer('https://i.imgur.com/VQorhER.jpg');

    // Fazendo decode da imagem e convertendo ela para o formato que devemos analisar
    const tensor = tf.node.decodeJpeg(buf, 3)
        .resizeNearestNeighbor([ 224, 224 ])
        .expandDims()
        .toFloat()
        .reverse(-1);
    // Executando o predict
    const data = await model.predict(tensor).data();

    // Aplicando as labels
    const result = applyLabels(data);

    console.log(result);
};

predict();