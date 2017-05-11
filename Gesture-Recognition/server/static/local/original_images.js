var CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"];

$(document).ready(function () {
    // Init
    initImages();
});

function initImages() {
    for (var i = 0; i < CLASSES.length; i++) {
        (function (i) {
            initImagesFromClass(CLASSES[i]);
        })(i);
    }
}

function initImagesFromClass(c) {
    ajaxGet('/api/image/list/original', {
        c: c
    }, function (res) {
        if (res.status === 0) {
            var $content = $('section.content');
            var $box = $('<div class="box box-default collapsed-box" id="' + c + '-images">  ' +
                '<div class="box-header with-border"> ' +
                '<h3 class="box-title">' + c + ' class Images Loading... </h3> ' +
                '<div class="box-tools pull-right"> ' +
                '<button class="btn btn-box-tool" data-widget="collapse"><i class="fa fa-plus"></i></button> ' +
                '</div><!-- /.box-tools --> ' +
                '</div><!-- /.box-header --> ' +
                '<div class="box-body"></div><!-- /.box-body --> ' +
                '</div><!-- /.box -->');

            var $boxTitle = $box.find('.box-title');
            var $boxBody = $box.find('.box-body');
            var $ul = $('<ul class="images" style="list-style: none;clear: both;"></ul>');
            var images = res.data, len = images.length, count = images.length;

            for (var i = 0; i < len; i++) {
                var image = images[i];
                var $li = $('<li class="image" style="position: relative;display: block;float: left;margin-right: 5px;margin-bottom: 5px;"></li>');
                var $img = $('<img class="o-item"' +
                    'data-base64="data:image/jpg;base64,' + image + '" src="data:image/jpg;base64,' + image + '" alt="" ' +
                    'style="cursor: pointer;display: block;width: 50px;height: 50px;">');

                $li.append($img);
                $ul.append($li);

                $img.on('load', function () {
                    if (--count === 0) {
                        $boxTitle.text($boxTitle.text() + 'done');
                    }
                });
            }

            $boxBody.append($ul);
            $content.append($box);

            // Listener
            $ul.on('click', 'li.image', function () {
                var $li = $(this);
                var $img = $li.find('img.o-item');
                var $img_processed = $li.find('img.p-item');

                if ($(this).data('clicked')) {
                    $img.css('display', 'block');
                    $img_processed.css('display', 'none');
                    $li.data('clicked', false);
                }
                else if ($img_processed && $img_processed.length) {
                    $img.css('display', 'none');
                    $img_processed.css('display', 'block');
                    $li.data('clicked', true);
                }
                else {
                    ajaxPost('/api/image/processing', {
                        image: $img.data('base64')
                    }, function (res) {
                        if (res.status === 0) {
                            var $img_processed = $('<img class="p-item"' +
                                'src="data:image/jpg;base64,' + res.data + '" alt="" ' +
                                'style="cursor: pointer;display: block;width: 50px;height: 50px;">');

                            $li.append($img_processed);
                            $li.data('clicked', true);
                            $img.css('display', 'none');
                        }
                    });
                }
            });
        }
    });
}

function createLoadingLayer() {
    var $loading = $('<div class="load8"> ' +
        '<div class="load8-container container1"> ' +
        '<div class="circle1"></div> ' +
        '<div class="circle2"></div> ' +
        '<div class="circle3"></div> ' +
        '<div class="circle4"></div> ' +
        '</div> ' +
        '<div class="load8-container container2"> ' +
        '<div class="circle1"></div> ' +
        '<div class="circle2"></div> ' +
        '<div class="circle3"></div> ' +
        '<div class="circle4"></div> ' +
        '</div> ' +
        '<div class="load8-container container3"> ' +
        '<div class="circle1"></div> ' +
        '<div class="circle2"></div> ' +
        '<div class="circle3"></div> ' +
        '<div class="circle4"></div> ' +
        '</div> ' +
        '</div>');

    return $loading;
}