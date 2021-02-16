$('.table').tablesorter();
    $('.sort1').on('change', function() {
        $('.table').tablesorter({
            sortList: [ this.value ? this.value.split(',') : [0, 0] ]
        });
    });

$('.table').tablesorter();
    $('.sort2').on('change', function() {
        $('.table').tablesorter({
            sortList: [ this.value ? this.value.split(',') : [0, 0] ]
        });
    });

$('.table').tablesorter();
    $('.sort3').on('change', function() {
        $('.table').tablesorter({
            sortList: [ this.value ? this.value.split(',') : [0, 0] ]
        });
    });

$('.table').tablesorter();
    $('.sort4').on('change', function() {
        $('.table').tablesorter({
            sortList: [ this.value ? this.value.split(',') : [0, 0] ]
        });
    });

$(function(){
    $('#className_list a').click(function (){
       var get_class = $(this).attr('class');
       $('.table thead').hide()
       $('.table tbody tr').hide()
       //console.log(get_class)
       $('.table thead').show()
       $('.table tbody tr[class="' + get_class + '"]').show()
    })
})

$(function () {
    $('.all_apply').click(function () {
        $('.table thead').hide()
       $('.table tbody tr').hide()
        $('.table thead').show()
       $('.table tbody tr').show()
    })
})