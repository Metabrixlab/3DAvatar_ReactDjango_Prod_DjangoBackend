var imageInput = document.getElementById('imageInput');
var preview = document.getElementById('preview');
var modified = document.getElementById('modified');
var flag = 0
imageInput.onchange = function(){
    preview.src = "media/original.png";
    modified.src = "media/modified.png";
    document.getElementById("process").style.opacity = "0";
    document.getElementById("modified").style.opacity = "1";

    document.getElementById("prev_img").innerHTML = "Preview image";
    flag = 0;
}

var bgInput = document.getElementById('bgInput');
var bg_preview = document.getElementById('bg_preview');
bgInput.onchange = function(){
    if (bgInput.files && bgInput.files[0]) {
        const reader = new FileReader();

        // Set the preview image source
        reader.onload = function (e) {
            bg_preview.src = e.target.result;
        }
        reader.readAsDataURL(bgInput.files[0]);
        preview.style.display = "block";
    }  
    else {
        bg_preview.src = "media/background.png";
    }
}

function previewImage() {
    const preview = document.getElementById('preview');
    const modified = document.getElementById('modified');
    const imageInput = document.getElementById('imageInput');

    // console.log(preview.src);
    // console.log(window.location.origin.concat('/media/original.png'));

    if(preview.src == window.location.origin.concat('/media/original.png'))
    {
        console.log("!!!!");
        console.log(imageInput.files[0]);
        
        // Make sure a file was selected
        if (imageInput.files && imageInput.files[0]) {
            const reader = new FileReader();

            // Set the preview image source
            reader.onload = function (e) {
                preview.src = e.target.result;
            }
            reader.readAsDataURL(imageInput.files[0]);

            // Show the preview image
            preview.style.display = "block";

            document.getElementById("prev_img").innerHTML = "Clear preview";
        }  
    }
    else
    {
        console.log("@@@@");
        preview.src = window.location.origin.concat('/media/original.png');
        modified.src = window.location.origin.concat('/media/modified.png');
        document.getElementById("process").style.opacity = "0";
        document.getElementById("modified").style.opacity = "1";

        document.getElementById("prev_img").innerHTML = "Preview image";
    }  
}


function submitImage1(bg_num, img_src) {
    const preview = document.getElementById('preview');
    const modified = document.getElementById('modified');
    const imageInput = document.getElementById('imageInput'); 
    const download_link1 = document.getElementById('down_link1'); 
    const download_link2 = document.getElementById('down_link2'); 


    if (imageInput.files && imageInput.files[0]) 
    {
        document.getElementById("process").style.opacity = "1";
        document.getElementById("modified").style.opacity = 0;
        // $("#modified").css("opacity", 0);
        // modified.src = window.location.origin.concat('/media/original.png');
        console.log(">>>", document.getElementById("modified").style.opacity);
        console.log(">>>", modified.src);

        if (flag == 0)
        {
            const reader = new FileReader();

            // Set the preview image source
            reader.onload = function (e) {preview.src = e.target.result;}
            reader.readAsDataURL(imageInput.files[0]);
            preview.style.display = "block";

            // Send the image file as the request body
            var formData = new FormData();
            var img_data = imageInput.files[0];

            console.log(img_data);
            console.log("temp");
            formData.append('image', img_data);
            formData.append('test', "temp");

            console.log("--------");
            for (var key of formData.entries()) {
                console.log(key[0] + ', ' + key[1]);
            }
            console.log("--------");

            $.ajax({
                url         : "/submit1/",
                type        : 'POST',
                data        : formData,
                // async       : false, 
                cache       : false,
                contentType : false,
                processData : false,
                success : function(response) {
                    console.log('success1');
                    // console.log(data);
                    // modified.src = 'data:image/png;base64,' + response.data;
                    // document.getElementById("process").style.opacity = "0";
                    // document.getElementById("modified").style.opacity = "1";

                    // download_link1.href = 'data:image/png;base64,' + response.data;
                    download_link2.href = 'data:image/png;base64,' + response.mask;
                    flag = 1;
                    change_back(bg_num, img_src);
                },
                error: function(data) {
                    console.log('image-fail1');
                }
            });
        }
        else {
            change_back(bg_num, img_src);
        }
         
    }
}

function change_back(bg_num, img_src) {
    const bgInput = document.getElementById('bgInput'); 
    const modified = document.getElementById('modified');
    const download_link1 = document.getElementById('down_link1'); 

    var formData = new FormData();
    formData.append('bg_num', bg_num);

    if (bg_num == 0)
    {
        if (bgInput.files && bgInput.files[0])
        { formData.append('img_src', bgInput.files[0]); }
        else
        { 
            modified.src = window.location.origin.concat('/media/modified.png');
            document.getElementById("process").style.opacity = "0";
            document.getElementById("modified").style.opacity = "1";

            return ;
        }
    }        
    else
    { formData.append('img_src', img_src); }
    

    console.log("--------");
    for (var key of formData.entries()) {
        console.log(key[0] + ', ' + key[1]);
    }
    console.log("--------");

    $.ajax({
        url         : "/change_bg/",
        type        : 'POST',
        // async       : false, 
        data        : formData,
        cache       : false,
        contentType : false,
        processData : false,
        success : function(response) {
            console.log('success2');
            // console.log(response.data);
            modified.src = 'data:image/png;base64,' + response.data;
            document.getElementById("process").style.opacity = "0";
            document.getElementById("modified").style.opacity = "1";

            download_link1.href = 'data:image/png;base64,' + response.data;
        },
        error: function(data) {
            console.log('image-fail2');
        }
    });  
}

function bg_func() {
    let buttons = document.querySelectorAll(".bg_img_but");
    console.log(buttons);
    buttons.forEach((btn, index) => {
        btn.addEventListener("click", bg_onclick, false);
    })
}

var bg_onclick = function(event) {
    console.log(event.target.src);
    console.log(this, this.value);
    console.log("?><?><")
    submitImage1(this.value, event.target.src);
};