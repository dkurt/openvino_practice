const lib = require('common');

function init() {
  let results = [];
  function add_task(name) {
    results.push({
      'name': name,
      'description': 'modules/' + name + '/README.md',
      'html_url': undefined,
      'status': 'neutral'
    });
  }
  add_task('0_git');
  add_task('1_opencv');
  add_task('2_mnist');
  add_task('3_classification');
  add_task('4_detection');
  add_task('5_tracking');
  add_task('6_segmentation');
  add_task('7_nlp');
  add_task('8_quantization');
  add_task('9_colorization');
  add_task('9_gan');
  add_task('10_reinforcement_learning');
  return results;
}

module.exports.checkAll = function checkAll(token, callback) {
  let results = {};
  lib.github_api_get('https://api.github.com/repos/dkurt/openvino_practice/pulls?per_page=100&sort=created', token, (pulls) => {
    var num = 0;
    function inc() {
      num += 1;
      if (num == pulls.length) {
        callback(results);
      }
    }

    pulls.forEach((pull) => {
      let username = pull.user.login;

      if (!(username in results)) {
        results[username] = init();
      }

      var moduleNames = [];
      var isAccepted = false;
      pull.labels.forEach((label) => {
        var matches = label.name.match(/module: (.*)/);
        if (matches) {
          moduleNames.push(matches[1]);
        }
        if (label.name.localeCompare("accepted") == 0) {
          isAccepted = true;
        }
      });
      moduleNames.forEach(function(moduleName) {
        var tasks = results[username];
        tasks.forEach((task) => {
          if (task.name.localeCompare(moduleName) == 0) {
            if (isAccepted) {
              task.status = 'success';
            } else {
              task.status = 'progress';
            }
            task.html_url = pull.html_url;
          }
        })
      });
      inc();
    });
  });
}

module.exports.check = function check(username, token, callback) {
  module.exports.checkAll(token, (results) => {
    let tasks = results[username];
    if (!tasks) {
      tasks = init();
    }
    callback(tasks);
  });
}
