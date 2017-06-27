var express = require('express')
var router = express.Router()

/* GET users listing. */
router.get('/', function(req, res, next) {
  res.send('respond with a resource')
})


var cmd = require('node-cmd')

router.get('/ml', function(req, res, next) {
  // Note: Activate TensorFlow before execute server.
  var pyCommand = 'python ../../model-comparator/ml/predict.py -i ../../model-comparator/ml/test_Keira.jpg'

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

module.exports = router
