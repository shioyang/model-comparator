import { Component } from '@angular/core'
import { ServerService } from './server.service'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'app'
  images = [
    {
      name: 'Pred #1',
      file_name: '1_pred.jpg',
      path: './ml/data_prediction/1_pred.jpg'
    },
    {
      name: 'Pred #2',
      file_name: '2_pred.jpg',
      path: './ml/data_prediction/2_pred.jpg'
    },
    {
      name: 'Pred #3',
      file_name: '3_pred.jpg',
      path: './ml/data_prediction/3_pred.jpg'
    }
  ]
  selected_image = ''
  result_text = ''
  isShowSpinner = false
  isShowFemale = false
  isShowMale = false

  constructor(private serverService: ServerService){}

  OnPredictClicked(): void {
    let image_path = this.selected_image.substring(5);
    console.log('image_path: ' + image_path)

    this.isShowSpinner = true
    this.isShowFemale = false
    this.isShowMale = false

    this.serverService.predictImage(image_path)
      .subscribe(result => {
        this.isShowSpinner = false
        this.result_text = result

        if(this.isFemale(result)){
          this.isShowSpinner = false
          this.isShowFemale = true
          this.isShowMale = false
        }else{
          // Male
          this.isShowSpinner = false
          this.isShowFemale = false
          this.isShowMale = true
        }
      })
  }

  isFemale(result_json){
    return (result_json.female > 50)
  }

}
