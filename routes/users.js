var express = require('express')
var router = express.Router()

/* GET users listing. */
router.get('/', function(req, res, next) {
  res.send('respond with a resource')
})

// http://localhost:3000/users/predict
router.get('/predict', function(req, res, next) {
  let file_path = req.query.file_path

  var spawn = require("child_process").spawn;
  var process = spawn('python', ["src/ml/predict.py", '-i', file_path]);
  process.stdout.setEncoding('utf8');
  console.log('spawned')

  process.stdout.on('data', function (data){
    var lines = data.split("\n");
    var result = lines[lines.length - 2]
    console.log('result: ' + result)
    res.send(result)
  });
})


module.exports = router