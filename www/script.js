window.onload = function() {
    setInterval(function(){
        var date = new Date();
        var displayDate = date.toLocaleDateString();
        var displayTime = date.toLocaleTimeString();

        document.getElementById('datetime').innerHTML = displayDate + " " + displayTime;
    }, 1000); // 1000 milliseconds = 1 second
}

function findFileWithCurrentDate() {
    const currentDate = new Date();
    const year = currentDate.getFullYear();
    const month = String(currentDate.getMonth() + 1).padStart(2, '0');
    const day = String(currentDate.getDate()).padStart(2, '0');
    const datePrefix = `${year}_${month}_${day}`;

    return `img/forecasts/Common Buzzard/${year}${month}${day}_Common_Buzzard.jpg`;
}

const fileFound = findFileWithCurrentDate();
document.write(`<img src="${fileFound}" alt="Common Buzzard">`);

// fetchFile(function(){
//     // const fileFound = findFileWithCurrentDate(files);
//     document.getElementById('myfile').innerHTML = "fileFound";
// }); // 1000 milliseconds = 1 second
