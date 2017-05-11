var DSSize = [130, 408, 999, 1989, 3186, 3966];
var scoreSVM = [0.704, 0.860, 0.913, 0.946, 0.968, 0.977];
var scoreKNN = [0.456, 0.687, 0.842, 0.921, 0.945, 0.969];
var scoreLogistic = [0.706, 0.864, 0.918, 0.946, 0.972, 0.978];

$(document).ready(function () {
    // Listeners
    $('#inputFile').on('change', function () {
        var $input = this;
        var $previewImage = document.getElementById('preview-image');
        var $processedImage = document.getElementById('processed-image');
        var $predicted = document.getElementsByClassName('predicted')[0];
        var $predictResult = document.getElementById('predict-result');
        var file = $input.files[0];
        var fr = new FileReader();

        if (!window.FileReader) {
            alert('Your browser not support FileReader');
            return;
        }

        if (/jpeg/.test(file.type)) {
            fr.readAsDataURL(file);
            fr.onloadend = function (e) {
                var image_base64 = e.target.result;

                $previewImage.style.display = 'block';
                $previewImage.src = image_base64;

                // Process image async(form base64)
                ajaxPost('/api/image/processing', {
                    image: image_base64
                }, function (res) {
                    if (res.status === 0) {
                        $processedImage.style.display = 'block';
                        $processedImage.src = 'data:image/jpg;base64,' + res.data;
                    }
                });

                // Predict image class
                ajaxPost('/api/image/predict', {
                    image: image_base64
                }, function (res) {
                    if (res.status === 0) {
                        $predicted.style.display = 'block';
                        $predictResult.innerText = res.data;
                    }
                });
            }
        }
        else {
            alert('Submit .jps images');
        }
    });

    // Init
    var data = {
        labels: DSSize,
        datasets: [
            {
                label: "SVM score",
                fillColor: "rgba(220,220,220,0.5)",
                strokeColor: "rgba(220,220,220,1)",
                pointColor: "rgba(220,220,220,1)",
                pointStrokeColor: "#fff",
                data: scoreSVM
            }, {
                label: "KNN score",
                fillColor: "rgba(151,187,205,0.5)",
                strokeColor: "rgba(151,187,205,1)",
                pointColor: "rgba(151,187,205,1)",
                pointStrokeColor: "#fff",
                data: scoreKNN
            }
        ]
    };

    var options = {
        bezierCurve: false,
        showScale: true,
        scaleShowGridLines: true
    };

    // Get context with jQuery - using jQuery's .get() method.
    var ctx = $("#chart").get(0).getContext("2d");
    // This will get the first returned node in the jQuery collection.
    var chart = new Chart(ctx).Line(data, options);
});