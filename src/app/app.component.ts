import { Component } from '@angular/core'

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
      path: './ml/data_prediction/1_pred.jpg'
    },
    {
      name: 'Pred #2',
      path: './ml/data_prediction/2_pred.jpg'
    },
    {
      name: 'Pred #3',
      path: './ml/data_prediction/3_pred.jpg'
    }
  ]

}
