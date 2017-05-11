function ajaxGet(url, param, callback) {
    $.ajax({
        type: 'GET',
        url: url,
        data: param,
        dataType: 'json',
        success: callback
    });
}

function ajaxPost(url, data, callback) {
    $.ajax({
        type: 'POST',
        url: url,
        data: JSON.stringify(data),
        dataType: 'json',
        contentType: 'application/json',
        success: callback
    });
}