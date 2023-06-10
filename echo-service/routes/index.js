var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/echo/:name', function(req, res, next) {
  res.json({message: `Welcome ${req.params.name}!`});
});

module.exports = router;
