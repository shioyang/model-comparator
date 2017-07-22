var express = require('express')
var router = express.Router()

/* GET users listing. */
router.get('/', function(req, res, next) {
  res.send('respond with a resource')
})


var cmd = require('node-cmd')

// http://localhost:3000/users/ml
router.get('/ml', function(req, res, next) {
  // Note: Activate TensorFlow before execute server.
  var pyCommand = 'cd ./src/ml && python predict.py -i data_prediction/2_pred.jpg'

  var pyProcess = cmd.get(pyCommand, function(data, err, stderr){
    if(!err){
      // ok
      res.send('Success: ' + data)
    }else{
      // error
      console.log('Error: ' + pyCommand)
      console.log('     : ' + stderr)
      res.send('Error: ' + pyCommand + '\n' + stderr)
    }
  })
})


var pysh = require('python-shell')

// http://localhost:3000/users/ml2
router.get('/ml2', function(req, res, next) {
  // Note: Activate TensorFlow before execute server.
  var pyCommand = 'predict.py'
  var options = {
    scriptPath: 'src/ml',
    args: ['-i','data_prediction/2_pred.jpg']
  }

  pysh.run(pyCommand, options, function(err, results){
    if(err){
      console.log('Error: ' + pyCommand)
      console.log('     : ' + err)
      console.log('?????: %j', results)
      res.send('Error: ' + pyCommand + '\n' + err)
    }else{
      // success
      console.log('Success: %j', results)
      res.send('Success: ' + results)
    }
  })
  
})


var request = require('request')

// http://localhost:3000/users/callpy
router.get('/callpy', function(req, res, next){
  request.get('http://localhost:5000/ml', function(error, response, body){
    res.send('body: ' + body)
  })
})

// http://localhost:3000/users/predict
router.get('/predict', function(req, res, next){
  let file_path = req.query.file_path
  console.log(file_path)
  let param = {
    url:  'http://localhost:5000/ml',
    formData: {
      file_path: file_path
    }
  }
  request.post(param, function(error, response, body){
    if(error){
      res.send('Error: ' + error)
    }else{
      // res.send('res statusCode: ' + response.statusCode)
      if(response.statusCode === 200){
        if(body)
          res.send(body)
        else
          res.send('body is empty.')
      }else{
        res.send('status code: ' + response.statusCode)
      }
    }
  })
})

module.exports = router
